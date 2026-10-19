"""CPU tests for v3.modal_hf: dequantization, checkpoint loading, projection math, hooks, prompt sets.

All prompt text is a placeholder. The one HF-backed test (tiny random Qwen3_5 model) needs transformers and is
skipped where it is not installed (the sprint .venv has none; run it with any env that has transformers 5.x).
"""
import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from v3 import modal_hf as mh


# ------------------------------------------------------------------------------------------ helpers
def quantize(w: torch.Tensor, block: int = 128):
    """Reference block-FP8 quantizer (per-block amax/448), mirroring the donors' scheme."""
    m, n = w.shape
    pm, pn = -(-m // block) * block, -(-n // block) * block
    padded = torch.zeros(pm, pn)
    padded[:m, :n] = w.float()
    tiles = padded.reshape(pm // block, block, pn // block, block).permute(0, 2, 1, 3)
    scale = tiles.abs().amax(dim=(-2, -1)) / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (tiles / scale[:, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.permute(0, 2, 1, 3).reshape(pm, pn)[:m, :n].contiguous(), scale


def dequant_loop(q: torch.Tensor, scale: torch.Tensor, block: int = 128):
    out = torch.zeros(q.shape)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            out[i * block:(i + 1) * block, j * block:(j + 1) * block] = (
                q[i * block:(i + 1) * block, j * block:(j + 1) * block].float() * scale[i, j].item())
    return out.to(torch.bfloat16)


class CharTok:
    """Stub tokenizer: one token per character (id = ord), pad id 0, records the template kwargs."""
    pad_id = 0
    padding_side = "right"

    def __init__(self, suffix="<think>\n"):
        self.suffix, self.calls = suffix, []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        eff = kw.get("reasoning_effort")
        return f"[{eff}]" + "|".join(m["content"] for m in messages) + self.suffix

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        ids = [torch.tensor([ord(c) % 250 + 1 for c in t]) for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for k, i in enumerate(ids):
            input_ids[k, :len(i)], mask[k, :len(i)] = i, 1
        return _Enc({"input_ids": input_ids, "attention_mask": mask})


class LeftPadTok(CharTok):
    """Same as CharTok but pads on the left: collect_activations must refuse such batches."""

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        enc = super().__call__(texts, return_tensors, padding, add_special_tokens)
        lens = enc["attention_mask"].sum(1)
        width = enc["input_ids"].shape[1]
        for k, n in enumerate(lens.tolist()):
            enc["input_ids"][k] = enc["input_ids"][k].roll(width - n)
            enc["attention_mask"][k] = enc["attention_mask"][k].roll(width - n)
        return enc


class _Enc(dict):
    def to(self, _device):
        return self


class ToyLayer(nn.Module):
    """Causal toy layer: h + W * running_mean(h). Counts calls so early stopping can be checked."""

    def __init__(self, d, as_tuple=False):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        self.calls, self.as_tuple = 0, as_tuple

    def forward(self, hidden_states):
        self.calls += 1
        t = hidden_states.shape[1]
        mean = hidden_states.cumsum(1) / torch.arange(1, t + 1)[None, :, None]
        out = hidden_states + self.lin(mean)
        return (out,) if self.as_tuple else out


class ToyInner(nn.Module):
    def __init__(self, d, n_layers, as_tuple, positional):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        self.layers = nn.ModuleList([ToyLayer(d, as_tuple) for _ in range(n_layers)])
        self.positional = positional

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            r = layer(h) if self.positional else layer(hidden_states=h)
            h = r[0] if isinstance(r, tuple) else r
        return h


class ToyLM(nn.Module):
    def __init__(self, d=16, n_layers=5, as_tuple=False, positional=True):
        super().__init__()
        self.model = ToyInner(d, n_layers, as_tuple, positional)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.model(input_ids)


def toy_texts():
    return ["[low]AAA|bb<think>\n", "[low]a much longer placeholder prompt|x<think>\n", "[low]z<think>\n", "[low]mid length|q<think>\n"]


# ------------------------------------------------------------------------------------------ dequantization
def test_dequantize_matches_block_loop_and_original():
    torch.manual_seed(0)
    w = torch.randn(256, 384) * torch.linspace(0.01, 3.0, 384)
    q, s = quantize(w)
    got = mh.dequantize_fp8_block(q, s)
    assert got.dtype == torch.bfloat16 and got.shape == w.shape
    assert torch.equal(got, dequant_loop(q, s))
    # e4m3 keeps ~3 mantissa bits: error is bounded by a fraction of each block's amax
    err = (got.float() - w).abs()
    amax = s.repeat_interleave(128, 0).repeat_interleave(128, 1) * 448.0
    assert float((err / amax).max()) < 0.07


def test_dequantize_non_divisible_shape():
    torch.manual_seed(1)
    w = torch.randn(200, 300)
    q, s = quantize(w)
    assert s.shape == (2, 3)
    got = mh.dequantize_fp8_block(q, s)
    assert got.shape == (200, 300)
    assert torch.allclose(got.float(), w, atol=0.35)


def test_dequantize_uses_scale_as_multiplier():
    q = torch.ones(128, 128).to(torch.float8_e4m3fn)
    got = mh.dequantize_fp8_block(q, torch.tensor([[2.5]]))
    assert torch.all(got.float() == 2.5)


def test_dequantize_rejects_bad_shapes():
    q = torch.ones(256, 128).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        mh.dequantize_fp8_block(q, torch.ones(1, 1))
    with pytest.raises(ValueError):
        mh.dequantize_fp8_block(q, torch.ones(2))


def test_canon_name():
    assert mh.canon_name("model.language_model.layers.3.mlp.down_proj.weight") == "layers.3.mlp.down_proj.weight"
    assert mh.canon_name("model.layers.3.mlp.down_proj.weight") == "layers.3.mlp.down_proj.weight"
    assert mh.canon_name("model.visual.blocks.0.attn.qkv.weight") == "visual.blocks.0.attn.qkv.weight"
    assert mh.canon_name("lm_head.weight") == "lm_head.weight"
    assert mh.canon_name("mtp.fc.weight") == "mtp.fc.weight"


# ------------------------------------------------------------------------------------------ checkpoint loading
class TinyNet(nn.Module):
    """HF text-only naming: model.embed_tokens / model.layers.N / model.norm / lm_head."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 128)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(256, 128, bias=False)
        layer.input_layernorm = nn.Module()
        layer.input_layernorm.weight = nn.Parameter(torch.ones(128))
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(128, 8, bias=False)
        self.to(torch.bfloat16)


def write_ckpt(tmp_path, drop=(), extra=None, no_scale=False, bad_shape=False, stray_scale=False, seed=2):
    torch.manual_seed(seed)
    w_down = torch.randn(128, 256)
    q, s = quantize(w_down)
    embed = torch.randn(8, 128).to(torch.bfloat16)
    tensors = {
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.0.mlp.down_proj.weight": q[:64] if bad_shape else q,
        "model.language_model.layers.0.input_layernorm.weight": torch.full((128,), 2.0).to(torch.bfloat16),
        "lm_head.weight": torch.randn(8, 128).to(torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.zeros(4, 4),
        "mtp.fc.weight": torch.zeros(4, 4),
    }
    if not no_scale:
        tensors["model.language_model.layers.0.mlp.down_proj.weight_scale_inv"] = s
    if stray_scale:
        tensors["lm_head.weight_scale_inv"] = torch.ones(1, 1)
    tensors.update(extra or {})
    lm_head = tensors["lm_head.weight"]
    for k in drop:
        tensors.pop(k)
    names = sorted(tensors)
    half = len(names) // 2
    files = {"a.safetensors": names[:half], "b.safetensors": names[half:]}
    index = {}
    for fname, keys in files.items():
        save_file({k: tensors[k].contiguous() for k in keys}, str(tmp_path / fname))
        index.update({k: fname for k in keys})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return {"down": dequant_loop(q, s), "embed": embed, "lm_head": lm_head}


def test_load_checkpoint_into_model_dequantizes_and_covers(tmp_path):
    expect = write_ckpt(tmp_path)
    net = TinyNet()
    stats = mh.load_checkpoint_into_model(net, str(tmp_path))
    assert stats == {"n_params": 4, "n_fp8_dequantized": 1, "n_copied": 3, "n_ignored": 2}
    assert torch.equal(net.model.layers[0].mlp.down_proj.weight.data, expect["down"])
    assert torch.equal(net.model.embed_tokens.weight.data, expect["embed"])
    assert torch.equal(net.lm_head.weight.data, expect["lm_head"])
    assert torch.all(net.model.layers[0].input_layernorm.weight.data == 2.0)


def test_load_checkpoint_missing_parameter_raises(tmp_path):
    write_ckpt(tmp_path, drop=["lm_head.weight"])
    with pytest.raises(KeyError, match="not found"):
        mh.load_checkpoint_into_model(TinyNet(), str(tmp_path))


def test_load_checkpoint_unexpected_tensor_raises(tmp_path):
    write_ckpt(tmp_path, extra={"model.language_model.layers.7.mlp.down_proj.weight": torch.zeros(2, 2)})
    with pytest.raises(KeyError, match="no model parameter"):
        mh.load_checkpoint_into_model(TinyNet(), str(tmp_path))


def test_load_checkpoint_fp8_without_scale_raises(tmp_path):
    write_ckpt(tmp_path, no_scale=True)
    with pytest.raises(KeyError, match="no model.language_model.layers.0.mlp.down_proj.weight_scale_inv"):
        mh.load_checkpoint_into_model(TinyNet(), str(tmp_path))


def test_load_checkpoint_scale_on_non_fp8_raises(tmp_path):
    write_ckpt(tmp_path, stray_scale=True)
    with pytest.raises(ValueError, match="not float8"):
        mh.load_checkpoint_into_model(TinyNet(), str(tmp_path))


def test_load_checkpoint_shape_mismatch_raises(tmp_path):
    write_ckpt(tmp_path, bad_shape=True)
    with pytest.raises(ValueError):
        mh.load_checkpoint_into_model(TinyNet(), str(tmp_path))


def test_check_config_matches_base(tmp_path):
    base, ck = tmp_path / "base", tmp_path / "ck"
    base.mkdir(), ck.mkdir()
    cfg = {"model_type": "x", "text_config": {"hidden_size": 5120}}
    (base / "config.json").write_text(json.dumps(cfg))
    (ck / "config.json").write_text(json.dumps({**cfg, "quantization_config": {"quant_method": "fp8"}}))
    mh.check_config_matches_base(str(ck), str(base))
    (ck / "config.json").write_text(json.dumps({**cfg, "text_config": {"hidden_size": 4096}}))
    with pytest.raises(ValueError, match="text_config"):
        mh.check_config_matches_base(str(ck), str(base))


# ------------------------------------------------------------------------------------------ vectors and projection
def write_vectors(tmp_path, seed=0, r1_best_layer=48):
    rng = np.random.default_rng(seed)
    unit = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=32))
    r1 = {k: unit() for k in mh.VECTOR_KEYS["r1"]}
    r2 = {k: unit() for k in mh.VECTOR_KEYS["r2"]}
    np.savez(tmp_path / "r1.npz", best_layer=np.array([r1_best_layer]), **r1)
    np.savez(tmp_path / "r2.npz", **r2)
    return {"r1": str(tmp_path / "r1.npz"), "r2": str(tmp_path / "r2.npz")}, r1, r2


def test_load_vectors_meta_and_cosines(tmp_path):
    files, r1, r2 = write_vectors(tmp_path)
    vecs, meta = mh.load_vectors(files)
    assert len(vecs) == 10 and set(vecs) == {f"r1.{k}" for k in mh.VECTOR_KEYS["r1"]} | {f"r2.{k}" for k in mh.VECTOR_KEYS["r2"]}
    assert meta["files"]["r1"]["best_layer"] == 48 and len(meta["files"]["r2"]["sha256"]) == 64
    assert meta["cos"]["r1.r_hat"]["r1.r_hat"] == pytest.approx(1.0)
    assert meta["cos"]["r1.r_hat"]["r2.refusal"] == pytest.approx(float(r1["r_hat"] @ r2["refusal"]))
    assert meta["cos"]["r2.refusal"]["r1.r_hat"] == meta["cos"]["r1.r_hat"]["r2.refusal"]


def test_load_vectors_rejects_non_unit(tmp_path):
    files, _, _ = write_vectors(tmp_path)
    np.savez(tmp_path / "r2.npz", **{k: np.ones(32) for k in mh.VECTOR_KEYS["r2"]})
    with pytest.raises(ValueError, match="unit-norm"):
        mh.load_vectors(files)


def test_projection_stats_known_values():
    v = np.zeros(4)
    v[0] = 1.0
    h = np.array([[3.0, 4.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    st = mh.projection_stats(h, {"x": v})
    assert st["n"] == 3 and st["h_norm"] == pytest.approx([5.0, 2.0, 1.0])
    assert st["vectors"]["x"]["dot"] == pytest.approx([3.0, -2.0, 0.0])
    assert st["vectors"]["x"]["mean_abs_ratio"] == pytest.approx((0.6 + 1.0 + 0.0) / 3)


def test_compare_acts_identical_and_perturbed():
    rng = np.random.default_rng(0)
    a = {"in": rng.normal(size=(5, 8)), "out": rng.normal(size=(5, 8))}
    same = mh.compare_acts(a, a)
    assert same["in"]["min_cos"] == pytest.approx(1.0) and same["out"]["max_rel_l2"] == pytest.approx(0.0)
    b = {k: v * 2 for k, v in a.items()}
    scaled = mh.compare_acts(b, a)
    assert scaled["in"]["min_cos"] == pytest.approx(1.0) and scaled["in"]["mean_rel_l2"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------ prompts
def write_donor_files(tmp_path):
    run1 = [{"id": f"val_{i:03d}", "split": "val", "modality": "agentic_tool" if i % 2 else "text",
             "label": "harmful" if i % 4 < 2 else "harmless", "category": "c", "tool_name": "t",
             "messages": [{"role": "user", "content": f"[HARM_TEXT_PLACEHOLDER_run1_{i:03d}]"}]} for i in range(64)]
    run2 = [{"trial_id": f"extraction_{i // 2:03d}_{'agentic' if i < 128 else 'text'}_{'harmful' if i % 2 else 'benign'}",
             "consequence": "harmful" if i % 2 else "harmless", "request_mode": "agentic" if i < 128 else "text",
             "messages": [{"role": "system", "content": "[SYS]"}, {"role": "user", "content": f"[HARM_TEXT_PLACEHOLDER_run2_{i:03d}]"}]}
            for i in range(256)]
    p1, p2 = tmp_path / "val.json", tmp_path / "extraction.jsonl"
    p1.write_text(json.dumps(run1))
    p2.write_text("".join(json.dumps(r) + "\n" for r in run2))
    return p1, p2


def test_prompt_sets_counts_balance_and_rows(tmp_path):
    p1, p2 = write_donor_files(tmp_path)
    prompts = mh.build_prompt_sets(p1, p2)
    assert len(prompts) == 128 and [p["set"] for p in prompts].count("r1_val") == 64
    r1 = [p for p in prompts if p["set"] == "r1_val"]
    r2 = [p for p in prompts if p["set"] == "r2_val"]
    assert r1[0]["modality"] in ("agentic", "text") and "_tool" not in "".join(p["modality"] for p in r1)
    assert len({p["id"] for p in prompts}) == 128
    assert len(mh.RUN2_VAL_ROWS) == 64 and len(set(mh.RUN2_VAL_ROWS)) == 64
    # literal expectation (pairs 48-63 agentic, 112-127 text), not derived from mh.RUN2_VAL_ROWS
    expected_rows = [2 * i + h for i in list(range(48, 64)) + list(range(112, 128)) for h in (0, 1)]
    assert [p["messages"][-1]["content"] for p in r2] == [f"[HARM_TEXT_PLACEHOLDER_run2_{i:03d}]" for i in expected_rows]
    # held-out pairs are 16 agentic + 16 text pairs, balanced by label
    assert sum(p["label"] == "harmful" for p in r2) == 32 and sum(p["modality"] == "agentic" for p in r2) == 32


def test_prompt_sets_reject_wrong_counts(tmp_path):
    p1, p2 = write_donor_files(tmp_path)
    p1.write_text(json.dumps(json.loads(p1.read_text())[:10]))
    with pytest.raises(ValueError):
        mh.load_run1_val(p1)
    p2.write_text(p2.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        mh.load_run2_val(p2)


def test_prompt_meta_has_no_text(tmp_path):
    p1, p2 = write_donor_files(tmp_path)
    meta = mh.prompt_meta(mh.build_prompt_sets(p1, p2))
    assert set(meta) == {"r1_val", "r2_val"} and len(meta["r1_val"]["messages_sha256"]) == 64
    assert "PLACEHOLDER" not in json.dumps(meta)


def test_head_per_set():
    prompts = [{"set": "a"}, {"set": "a"}, {"set": "a"}, {"set": "b"}, {"set": "b"}]
    assert mh.head_per_set(prompts, 2) == [0, 1, 3, 4]
    assert mh.head_per_set(prompts, 1) == [0, 3]
    assert mh.head_per_set(prompts, 9) == [0, 1, 2, 3, 4]


def test_render_prompt_effort_kwargs_and_think_suffix():
    tok = CharTok()
    msgs = [{"role": "user", "content": "[HARM_TEXT_PLACEHOLDER]"}]
    assert mh.render_prompt(tok, msgs, "low").endswith("<think>\n")
    mh.render_prompt(tok, msgs, None)
    mh.render_prompt(tok, msgs, "default")
    assert tok.calls[0] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True, "reasoning_effort": "low"}
    assert "reasoning_effort" not in tok.calls[1] and "reasoning_effort" not in tok.calls[2]
    with pytest.raises(ValueError, match="think"):
        mh.render_prompt(CharTok(suffix="\n"), msgs, "low")


# ------------------------------------------------------------------------------------------ hooks
@pytest.mark.parametrize("as_tuple,positional", [(False, True), (True, True), (False, False)])
def test_collect_activations_matches_manual_forward_and_stops_early(as_tuple, positional):
    torch.manual_seed(0)
    model, tok, texts = ToyLM(as_tuple=as_tuple, positional=positional), CharTok(), toy_texts()
    layers = mh.find_layers(model)
    got = mh.collect_activations(model, layers, tok, texts, layer=2, batch_size=3, device="cpu")
    assert [layers[i].calls for i in range(5)] == [2, 2, 2, 0, 0]          # 4 texts in batches of 3 + 1; layers above 2 never run
    for k, text in enumerate(texts):
        ids = tok([text])["input_ids"]
        h = model.model.embed(ids)
        for i in range(2):
            h = layers[i](h) if positional else layers[i](hidden_states=h)
            h = h[0] if as_tuple else h
        out = layers[2](h)
        out = out[0] if as_tuple else out
        assert torch.allclose(torch.tensor(got["in"][k]), h[0, -1], atol=1e-6)
        assert torch.allclose(torch.tensor(got["out"][k]), out[0, -1], atol=1e-6)
        assert got["last_ids"][k] == ord("\n") % 250 + 1 and got["n_tokens"][k] == len(text)
    assert all(not l._forward_hooks and not l._forward_pre_hooks for l in layers)   # hooks removed


def test_collect_activations_batching_equals_single():
    torch.manual_seed(1)
    model, tok, texts = ToyLM(n_layers=4), CharTok(), toy_texts()
    layers = mh.find_layers(model)
    batched = mh.collect_activations(model, layers, tok, texts, 3, 4, "cpu")
    single = mh.collect_activations(model, layers, tok, texts, 3, 1, "cpu")
    cmp = mh.compare_acts(batched, single)
    assert cmp["in"]["max_rel_l2"] < 1e-6 and cmp["out"]["max_rel_l2"] < 1e-6


def test_collect_activations_layer_out_of_range_never_fires():
    model = ToyLM(n_layers=2)
    with pytest.raises(IndexError):
        mh.collect_activations(model, mh.find_layers(model), CharTok(), toy_texts(), 5, 2, "cpu")


def test_find_layers_both_layouts():
    text_only = ToyLM()
    assert mh.find_layers(text_only) is text_only.model.layers
    multi = ToyLM()
    multi.model.language_model = nn.Module()
    multi.model.language_model.layers = nn.ModuleList([ToyLayer(16)])
    assert mh.find_layers(multi) is multi.model.language_model.layers


def test_evaluate_loaded_model_zero_projection_on_removed_direction():
    torch.manual_seed(3)
    model, tok = ToyLM(d=16, n_layers=4), CharTok()
    prompts = [{"id": f"p{i}", "set": "r1_val" if i < 3 else "r2_val", "label": "harmful", "modality": "text",
                "messages": [{"role": "user", "content": f"[HARM_TEXT_PLACEHOLDER_{i}]" + "x" * i}]} for i in range(6)]
    acts = mh.collect_activations(model, mh.find_layers(model), tok, [mh.render_prompt(tok, p["messages"], "low") for p in prompts], 2, 4, "cpu")
    rng = np.random.default_rng(0)
    r = rng.normal(size=16)
    r /= np.linalg.norm(r)
    results, got_acts, checks = mh.evaluate_loaded_model(model, tok, prompts, ["low", "default"], {"v": r}, layer=2, batch_size=4, device="cpu")
    assert set(results) == {"low", "default"} and set(results["low"]) == {"r1_val", "r2_val"}
    st = results["low"]["r1_val"]["out"]
    assert st["n"] == 3 and len(st["vectors"]["v"]["dot"]) == 3
    expect = np.abs(acts["out"][:3].astype(np.float64) @ r) / np.linalg.norm(acts["out"][:3].astype(np.float64), axis=1)
    assert st["vectors"]["v"]["mean_abs_ratio"] == pytest.approx(expect.mean())
    assert got_acts["low"]["in"].shape == (6, 16)
    assert checks["low"]["rendered_sha256"] != checks["default"]["rendered_sha256"]     # effort changes the prompt text
    assert sum(checks["low"]["last_token_ids"].values()) == 6 and len(checks["low"]["last_token_ids"]) == 1
    # project r out of the layer's output direction after the fact: ratio must vanish
    h = got_acts["low"]["out"].astype(np.float64)
    h_abl = h - np.outer(h @ r, r)
    assert mh.projection_stats(h_abl, {"v": r})["vectors"]["v"]["mean_abs_ratio"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------------------------ matrix and tables
def test_checkpoint_matrix_and_selection():
    matrix = mh.checkpoint_matrix()
    names = [m["name"] for m in matrix]
    assert len(names) == 13 and len(set(names)) == 13 and names[0] == "bf16_base"
    assert {n for n in names if n.startswith("r1_")} == {f"r1_{a}" for a in ("intact", "refusal_ablated", "pca_101", "pca_102", "pca_103", "unrelated")}
    assert all(m["path"].startswith("/weights/") for m in matrix)
    assert [m["name"] for m in mh.select_checkpoints("smoke")] == ["bf16_base", "r1_intact", "r1_refusal_ablated"]
    assert len(mh.select_checkpoints("all")) == 13
    assert [m["name"] for m in mh.select_checkpoints("r2_intact, official_fp8")] == ["r2_intact", "official_fp8"]
    with pytest.raises(ValueError, match="unknown"):
        mh.select_checkpoints("r3_intact")


def test_split_shards():
    assert mh.split_shards(list("abcdefg"), 3) == [list("adg"), list("be"), list("cf")]
    assert mh.split_shards(list("ab"), 5) == [["a"], ["b"]]
    assert mh.split_shards(list("abc"), 1) == [list("abc")]


def test_flatten_means_reductions():
    def rec(v):
        st = {"vectors": {"r1.r_hat": {"mean_abs_ratio": v}}}
        return {"results": {"low": {"r1_val": {"out": st, "in": st}}}}
    rows = mh.flatten_means({"bf16_base": rec(0.10), "r1_intact": rec(0.08), "r1_refusal_ablated": rec(0.008), "r2_intact": rec(0.5)})
    by = {r["ckpt"]: r for r in rows if r["site"] == "out"}
    assert by["r1_refusal_ablated"]["reduction_vs_intact"] == pytest.approx(0.9)
    assert by["r1_refusal_ablated"]["reduction_vs_base"] == pytest.approx(0.92)
    assert by["bf16_base"]["reduction_vs_intact"] is None and by["bf16_base"]["reduction_vs_base"] == pytest.approx(0.0)
    assert by["r2_intact"]["reduction_vs_intact"] == pytest.approx(0.0)


def test_modal_app_wiring():
    assert mh.app.name == "agent-safety-v3-20261018-projection"
    assert mh.OUT_DIR.startswith("/artifacts/agent-safety-v3-20261018/") and mh.LAYER == 48
    assert callable(mh.project_checkpoints.map)


# ------------------------------------------------------------------------------------------ real HF architecture (optional)
def test_tiny_qwen3_5_end_to_end_with_fp8_checkpoint(tmp_path):
    """Random tiny Qwen3_5ForCausalLM -> synthetic FP8 checkpoint with real checkpoint naming -> load into a
    blank copy -> identical layer-2 activations. Also checks hidden_states[2]/[3] are the layer-2 input/output."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model("qwen3_5")
    tc = cfg.text_config
    tc.num_hidden_layers, tc.layer_types = 4, ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.hidden_size, tc.intermediate_size, tc.vocab_size = 128, 256, 256
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 32
    for k, v in (("linear_num_key_heads", 2), ("linear_num_value_heads", 4), ("linear_key_head_dim", 16), ("linear_value_head_dim", 16)):
        setattr(tc, k, v)
    torch.manual_seed(0)
    src = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    tensors = {}
    for name, p in src.state_dict().items():
        key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        if name.endswith("proj.weight") and p.ndim == 2 and p.shape[0] % 128 == 0 and p.shape[1] % 128 == 0:
            q, s = quantize(p.float())
            tensors[key], tensors[key + "_scale_inv"] = q, s
        else:
            tensors[key] = p.contiguous()
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(2, 2)
    save_file(tensors, str(tmp_path / "all.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "all.safetensors" for k in tensors}}))

    blank = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    stats = mh.load_checkpoint_into_model(blank, str(tmp_path))
    assert stats["n_fp8_dequantized"] > 0 and stats["n_ignored"] == 1 and stats["n_params"] == len(list(blank.parameters()))

    tok = CharTok()
    texts = toy_texts()
    a = mh.collect_activations(blank, mh.find_layers(blank), tok, texts, 2, 3, "cpu")
    ids = tok(texts[:1])["input_ids"]
    with torch.no_grad():
        hs = blank(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    n = ids.shape[1]
    assert torch.allclose(torch.tensor(a["in"][0]), hs[2][0, n - 1].float(), atol=1e-2)    # layer-2 input = hidden_states[2]
    assert torch.allclose(torch.tensor(a["out"][0]), hs[3][0, n - 1].float(), atol=1e-2)   # layer-2 output = hidden_states[3]
    # the dequantized model is close to (not equal to) the source: FP8 rounding only
    ref = mh.collect_activations(src, mh.find_layers(src), tok, texts, 2, 3, "cpu")
    assert mh.compare_acts(a, ref)["out"]["min_cos"] > 0.98


# ------------------------------------------------------------------------------------------ launcher (scripts/run_projection.py)
def load_launcher():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_projection.py"
    spec = importlib.util.spec_from_file_location("run_projection", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_shard_output(tmp_path, prompts):
    """Run the toy model through evaluate_loaded_model for three named 'checkpoints' to get realistic records."""
    files, _, _ = write_vectors(tmp_path)
    vectors, meta = mh.load_vectors(files)
    tok = CharTok()
    ckpts = {}
    base_acts = None
    for k, name in enumerate(["bf16_base", "r1_intact", "r1_refusal_ablated"]):
        torch.manual_seed(5 + (k == 2))
        model = ToyLM(d=32, n_layers=4)
        results, acts, checks = mh.evaluate_loaded_model(model, tok, prompts, ["low"], vectors, layer=2, batch_size=4, device="cpu")
        rec = {"name": name, "results": results, "checks": checks, "load": {}}
        if base_acts is not None:
            rec["rel_l2_vs_bf16_base"] = {"low": mh.compare_acts(acts["low"], base_acts)}
        base_acts = base_acts if base_acts is not None else acts["low"]
        ckpts[name] = rec
    return {"shard_id": 0, "versions": {"torch": "x"}, "vectors": meta, "ckpts": ckpts, "elapsed_s": 12.0,
            "batch_check": {"batch_size": 1, "alt_batch_size": 8, "n_prompts": 8, "agreement": {"in": {}, "out": {}}}}


def test_launcher_dry_run_prints_counts_only(tmp_path, capsys):
    p1, p2 = write_donor_files(tmp_path)
    launcher = load_launcher()
    assert launcher.main(["--tag", "t", "--ckpts", "smoke", "--shards", "2", "--dry-run",
                          "--run1-prompts", str(p1), "--run2-prompts", str(p2)]) == 0
    out = capsys.readouterr().out
    assert "prompts=128" in out and "r1_val: 64" in out and "PLACEHOLDER" not in out
    assert "[['bf16_base', 'r1_refusal_ablated'], ['r1_intact']]" in out


def test_launcher_make_jobs_build_record_and_summary(tmp_path, capsys):
    launcher = load_launcher()
    p1, p2 = write_donor_files(tmp_path)
    prompts = mh.build_prompt_sets(p1, p2)
    toy_prompts = [{**p, "messages": [{"role": "user", "content": "x" * (i % 7 + 1)}]} for i, p in enumerate(prompts[:6] + prompts[64:70])]
    args = launcher.argparse.Namespace(tag="t", efforts="low", shards=2, layer=2, batch_size=4, no_crosscheck=False)
    jobs = launcher.make_jobs(args, prompts, mh.select_checkpoints("smoke"))
    assert [j["shard_id"] for j in jobs] == [0, 1] and jobs[0]["ckpts"][0]["name"] == "bf16_base"
    assert all(j["layer"] == 2 and j["efforts"] == ["low"] and j["crosscheck"] for j in jobs)
    out = fake_shard_output(tmp_path, toy_prompts)
    rec = launcher.build_record(args, prompts, [out], wall_s=60.0, est_usd=0.5)
    assert rec["schema"] == "projection_v1" and set(rec["checkpoints"]) == {"bf16_base", "r1_intact", "r1_refusal_ablated"}
    assert rec["vectors"]["cos"]["r1.r_hat"]["r2.refusal"] == out["vectors"]["cos"]["r1.r_hat"]["r2.refusal"]
    assert "PLACEHOLDER" not in json.dumps(rec) and rec["prompt_sets"]["r2_val"]["ids"][0].startswith("extraction_")
    json.dumps(rec)
    launcher.print_summary(rec)
    text = capsys.readouterr().out
    assert "cos(r1.r_hat, r2.refusal)" in text and "effort=low site=layer2_out" in text and "r1_refusal_ablated" in text
    assert "PLACEHOLDER" not in text


# ================================================================================================ reviewer additions
# Adversarial tests added in the independent review: each one targets a way the module (or a paid run) could be
# wrong without any of the earlier tests noticing (all prompt text is placeholder).

# ------------------------------------------------------------------------------------------ held-out split
DONOR_RUN2_EXTRACTION = pathlib.Path(__file__).resolve().parents[2] / "agent-safety-clean-v2-20261017/stimuli/phase1_extraction.jsonl"


def test_run2_validation_rows_are_exactly_the_donors_held_out_pairs():
    """Oracle copied from the donor's prepare_directions.py split (96 train pairs / 32 validation pairs), not from mh."""
    train_pairs = list(range(48)) + list(range(64, 112))
    val_pairs = list(range(48, 64)) + list(range(112, 128))
    assert sorted(train_pairs + val_pairs) == list(range(128))
    train_rows = {2 * i + h for i in train_pairs for h in (0, 1)}
    assert mh.RUN2_VAL_ROWS == [2 * i + h for i in val_pairs for h in (0, 1)]
    assert not train_rows & set(mh.RUN2_VAL_ROWS)      # nothing the r_hat was fit on


@pytest.mark.skipif(not DONOR_RUN2_EXTRACTION.exists(), reason="Run 2 donor sprint not on disk")
def test_real_run2_validation_prompts_are_held_out_pairs_with_opposite_labels():
    """Real donor file, structural fields only: assertions never touch a prompt dict, so a failure cannot print prompt text."""
    meta = [(int(p["id"].split("_")[1]), p["label"], p["modality"]) for p in mh.load_run2_val(DONOR_RUN2_EXTRACTION)]
    assert sorted({m[0] for m in meta}) == list(range(48, 64)) + list(range(112, 128))
    assert [m[2] for m in meta if m[0] < 64] == ["agentic"] * 32 and [m[2] for m in meta if m[0] >= 64] == ["text"] * 32
    pairs = list(zip(meta[0::2], meta[1::2]))
    assert all(a[0] == b[0] and {a[1], b[1]} == {"harmless", "harmful"} for a, b in pairs) and len(pairs) == 32


# ------------------------------------------------------------------------------------------ dequantization edge cases
def test_dequantize_accepts_bf16_scales_and_zero_blocks():
    q = torch.full((256, 128), 3.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [0.0]], dtype=torch.bfloat16)        # donors store bf16 scales; one dead block
    got = mh.dequantize_fp8_block(q, scale)
    assert torch.all(got[:128].float() == 1.5) and torch.all(got[128:].float() == 0.0)
    with pytest.raises(ValueError):
        mh.dequantize_fp8_block(q, torch.ones(3, 1))                    # too many scale rows is also a mismatch
    with pytest.raises(ValueError):
        mh.dequantize_fp8_block(q.reshape(2, 128, 128), torch.ones(2, 1))


# ------------------------------------------------------------------------------------------ loader semantics
def test_swapping_checkpoints_leaves_no_trace_of_the_previous_one(tmp_path):
    """One model object is reused for all 13 checkpoints: loading B after A must reproduce B exactly."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    exp_a, exp_b = write_ckpt(a, seed=2), write_ckpt(b, seed=3)
    net = TinyNet()
    mh.load_checkpoint_into_model(net, str(a))
    assert torch.equal(net.model.layers[0].mlp.down_proj.weight.data, exp_a["down"])
    mh.load_checkpoint_into_model(net, str(b))
    assert torch.equal(net.model.layers[0].mlp.down_proj.weight.data, exp_b["down"])
    assert torch.equal(net.model.embed_tokens.weight.data, exp_b["embed"])
    assert torch.equal(net.lm_head.weight.data, exp_b["lm_head"])
    assert not torch.equal(exp_a["down"], exp_b["down"]) and not torch.equal(exp_a["embed"], exp_b["embed"])   # A and B really differ


def test_load_checkpoint_rejects_unscaled_exotic_dtypes(tmp_path):
    """A float8_e5m2 / int8 tensor without a scale would otherwise be copied as garbage numbers."""
    for dtype in (torch.float8_e5m2, torch.int8):
        d = tmp_path / str(dtype)
        d.mkdir()
        write_ckpt(d, extra={"lm_head.weight": torch.ones(8, 128).to(dtype)})
        with pytest.raises(ValueError, match="unsupported dtype"):
            mh.load_checkpoint_into_model(TinyNet(), str(d))


def test_load_checkpoint_rejects_colliding_canonical_names(tmp_path):
    write_ckpt(tmp_path)
    net = TinyNet()
    net.model.language_model = nn.Module()
    net.model.language_model.embed_tokens = nn.Embedding(8, 128).to(torch.bfloat16)   # canonicalizes to the same key as model.embed_tokens
    with pytest.raises(ValueError, match="share canonical name"):
        mh.load_checkpoint_into_model(net, str(tmp_path))


# ------------------------------------------------------------------------------------------ hooks: failure modes
def test_collect_activations_rejects_left_padded_batches():
    model = ToyLM()
    with pytest.raises(ValueError, match="right-padded"):
        mh.collect_activations(model, mh.find_layers(model), LeftPadTok(), toy_texts(), 2, 4, "cpu")
    # batch_size=1 never pads, so the same tokenizer is fine there
    mh.collect_activations(model, mh.find_layers(model), LeftPadTok(), toy_texts(), 2, 1, "cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_collect_activations_is_equivariant_to_input_order(batch_size):
    torch.manual_seed(4)
    model, tok, texts = ToyLM(n_layers=4), CharTok(), toy_texts()
    layers, perm = mh.find_layers(model), [2, 0, 3, 1]
    base = mh.collect_activations(model, layers, tok, texts, 3, batch_size, "cpu")
    shuffled = mh.collect_activations(model, layers, tok, [texts[i] for i in perm], 3, batch_size, "cpu")
    for k, i in enumerate(perm):
        for site in mh.SITES:
            assert np.allclose(shuffled[site][k], base[site][i], atol=1e-6)
        assert shuffled["n_tokens"][k] == base["n_tokens"][i] == len(texts[i])


def test_collect_activations_errors_propagate_and_hooks_are_removed():
    class Boom(ToyLayer):
        def forward(self, hidden_states):
            raise RuntimeError("boom")

    model = ToyLM(n_layers=4)
    model.model.layers[1] = Boom(16)
    layers = mh.find_layers(model)
    with pytest.raises(RuntimeError, match="boom"):                       # not swallowed, not mistaken for the early stop
        mh.collect_activations(model, layers, CharTok(), toy_texts(), 3, 2, "cpu")
    assert all(not l._forward_hooks and not l._forward_pre_hooks for l in layers)


def test_collect_activations_raises_when_the_hooked_layer_never_runs():
    class SkipLM(ToyLM):
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            h = self.model.embed(input_ids)
            return self.model.layers[0](h)                                # layers 1-3 are never called

    model = SkipLM(n_layers=4)
    with pytest.raises(RuntimeError, match="never called"):
        mh.collect_activations(model, mh.find_layers(model), CharTok(), toy_texts(), 3, 2, "cpu")
    assert all(not l._forward_hooks and not l._forward_pre_hooks for l in mh.find_layers(model))


def test_collect_activations_rejects_degenerate_arguments():
    model, tok = ToyLM(), CharTok()
    for bs, texts in ((0, toy_texts()), (-1, toy_texts()), (2, [])):
        with pytest.raises(ValueError):
            mh.collect_activations(model, mh.find_layers(model), tok, texts, 2, bs, "cpu")


# ------------------------------------------------------------------------------------------ prompts, efforts, selection
def test_parse_efforts_and_render_validation():
    assert mh.parse_efforts("low,default") == ["low", "default"]
    assert mh.parse_efforts(" low , xhigh ") == ["low", "xhigh"]
    for bad in ("", "low,", "low,low", "Low", "high", "low;default"):
        with pytest.raises(ValueError):
            mh.parse_efforts(bad)
    with pytest.raises(ValueError, match="unknown reasoning effort"):
        mh.render_prompt(CharTok(), [{"role": "user", "content": "x"}], "high")


def test_render_prompt_error_never_echoes_prompt_text():
    msgs = [{"role": "user", "content": "[HARM_TEXT_PLACEHOLDER_secret]"}]
    with pytest.raises(ValueError) as err:
        mh.render_prompt(CharTok(suffix="[HARM_TEXT_PLACEHOLDER_tail]"), msgs, "low")
    assert "PLACEHOLDER" not in str(err.value) and "think" in str(err.value)


def test_prompts_with_forbidden_literals_or_duplicate_ids_are_rejected(tmp_path):
    """PLAN rule 5: no '</think>' and no DECLARED_TOOL_NAME in any prompt; errors name ids, never prompt text."""
    p1, p2 = write_donor_files(tmp_path)
    rows = json.loads(p1.read_text())
    rows[5]["messages"][0]["content"] += " [HARM_TEXT_PLACEHOLDER_x] </think> tail"
    p1.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="val_005") as err:
        mh.build_prompt_sets(p1, p2)
    assert "PLACEHOLDER" not in str(err.value)
    rows[5]["messages"][0]["content"] = '<tool_call>{"name": "DECLARED_TOOL_NAME"}</tool_call>'
    p1.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="forbidden"):
        mh.build_prompt_sets(p1, p2)
    rows[5]["messages"][0]["content"] = "[HARM_TEXT_PLACEHOLDER_ok]"
    rows[6]["id"] = rows[7]["id"]
    p1.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate"):
        mh.build_prompt_sets(p1, p2)
    # the rendered prompt is checked too (a template that injected a closed think block would be caught)
    with pytest.raises(ValueError, match="forbidden"):
        mh.render_prompt(CharTok(suffix="</think>\n<think>\n"), [{"role": "user", "content": "x"}], "low")


def test_select_checkpoints_rejects_empty_and_duplicate_selections():
    for bad in ("", " , ", "r1_intact,r1_intact"):
        with pytest.raises(ValueError):
            mh.select_checkpoints(bad)


def test_flatten_means_reference_missing_or_zero_gives_none():
    def rec(v):
        st = {"vectors": {"v": {"mean_abs_ratio": v}}}
        return {"results": {"low": {"s": {"out": st}}}}
    rows = {r["ckpt"]: r for r in mh.flatten_means({"r1_refusal_ablated": rec(0.1), "official_fp8": rec(0.1), "r2_intact": rec(0.0),
                                                     "r2_pca_101": rec(0.2)})}
    assert rows["r1_refusal_ablated"]["reduction_vs_intact"] is None and rows["r1_refusal_ablated"]["reduction_vs_base"] is None
    assert rows["official_fp8"]["reduction_vs_intact"] is None
    assert rows["r2_pca_101"]["reduction_vs_intact"] is None            # reference ratio is exactly 0: no division by zero


# ------------------------------------------------------------------------------------------ the Modal job body, on CPU
class _Vol:
    def reload(self):
        pass

    def commit(self):
        pass


def _toy(seed):
    torch.manual_seed(seed)
    return ToyLM(d=32, n_layers=4)


def run_job_on_cpu(tmp_path, monkeypatch, ckpt_names, batch_size=1, hf_fails=False, crosscheck=True):
    """Run the body of project_checkpoints with the weights, tokenizer, volumes and HF classes replaced by toys."""
    files, _, _ = write_vectors(tmp_path)
    seeds = {"bf16_base": 10, "r1_intact": 11, "r1_refusal_ablated": 12}
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "stub"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return CharTok()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kw):
            if hf_fails:
                raise OSError("no such model")
            return _toy(seeds["bf16_base"])

    fake_tf.AutoTokenizer, fake_tf.AutoModelForCausalLM = AutoTokenizer, AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(mh, "VECTOR_FILES", files)
    monkeypatch.setattr(mh, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mh, "art_vol", _Vol())
    monkeypatch.setattr(mh, "build_model", lambda base_dir, device: _toy(99))
    monkeypatch.setattr(mh, "check_config_matches_base", lambda *a, **k: None)
    monkeypatch.setattr(mh, "load_checkpoint_into_model",
                        lambda model, path: model.load_state_dict(_toy(seeds[path]).state_dict()) and {})
    prompts = [{"id": f"p{i}", "set": "r1_val" if i < 6 else "r2_val", "label": "harmless", "modality": "text",
                "messages": [{"role": "user", "content": "[HARM_TEXT_PLACEHOLDER]" + "z" * i}]} for i in range(12)]
    job = {"tag": "t", "shard_id": 0, "ckpts": [{"name": n, "path": n} for n in ckpt_names], "prompts": prompts,
           "efforts": ["low", "default"], "layer": 2, "batch_size": batch_size, "crosscheck": crosscheck}
    return mh.project_checkpoints.local(job), prompts, files


def test_project_checkpoints_body_on_cpu(tmp_path, monkeypatch):
    result, prompts, files = run_job_on_cpu(tmp_path, monkeypatch, ["r1_intact", "bf16_base", "r1_refusal_ablated"])
    assert list(result["ckpts"]) == ["bf16_base", "r1_intact", "r1_refusal_ablated"]       # bf16_base moved to the front
    base, intact = result["ckpts"]["bf16_base"], result["ckpts"]["r1_intact"]
    assert result["hf_reference"]["class"] == "ToyLM"
    assert base["swap_vs_hf_from_pretrained"]["out"]["max_rel_l2"] == 0.0                   # same weights through both paths
    assert "rel_l2_vs_bf16_base" not in base and set(intact["rel_l2_vs_bf16_base"]) == {"low", "default"}
    assert intact["rel_l2_vs_bf16_base"]["low"]["out"]["mean_rel_l2"] > 0.01              # different weights really differ
    assert result["batch_check"]["batch_size"] == 1 and result["batch_check"]["alt_batch_size"] == 8
    assert result["batch_check"]["n_prompts"] == 8 and result["batch_check"]["agreement"]["out"]["max_rel_l2"] < 1e-6
    assert all(rec["prompts_identical_to_first_ckpt"] for rec in result["ckpts"].values())
    json.dumps(result)                                                                      # the shard result must be serializable
    # files on the 'volume': the record and the activations, and the saved activations reproduce the record's numbers
    out = tmp_path / "out" / "t"
    assert json.loads((out / "r1_intact.json").read_text())["results"] == json.loads(json.dumps(intact["results"]))
    z = np.load(out / "acts_r1_intact.npz")
    assert set(z.files) == {"prompt_ids", "low__in", "low__out", "default__in", "default__out"} and list(z["prompt_ids"]) == [p["id"] for p in prompts]
    vectors, _ = mh.load_vectors(files)
    again = mh.projection_stats(z["default__in"][6:], vectors)
    assert again["vectors"]["r2.refusal"]["mean_abs_ratio"] == pytest.approx(
        intact["results"]["default"]["r2_val"]["in"]["vectors"]["r2.refusal"]["mean_abs_ratio"])


def test_project_checkpoints_survives_a_failed_hf_reference_and_reports_it(tmp_path, monkeypatch):
    result, _, _ = run_job_on_cpu(tmp_path, monkeypatch, ["bf16_base", "r1_intact"], batch_size=4, hf_fails=True)
    assert "OSError" in result["hf_reference"]["error"] and "swap_vs_hf_from_pretrained" not in result["ckpts"]["bf16_base"]
    assert result["batch_check"]["batch_size"] == 4 and result["batch_check"]["alt_batch_size"] == 1
    launcher = load_launcher()
    args = launcher.argparse.Namespace(tag="t", efforts="low,default", shards=1, layer=2, batch_size=4, no_crosscheck=False)
    p1, p2 = write_donor_files(tmp_path)
    record = launcher.build_record(args, mh.build_prompt_sets(p1, p2), [result], 1.0, 0.0)
    assert any("cross-check FAILED" in f for f in record["flags"])


# ------------------------------------------------------------------------------------------ launcher: money and flags
def launcher_with_fakes(tmp_path, monkeypatch, run_modal):
    """Launcher module whose ledger, Modal call and record directory are all fakes (the real ledger is never written)."""
    launcher = load_launcher()
    calls = []
    monkeypatch.setattr(launcher.ledger, "reserve", lambda job, usd, gpus=1: calls.append(("reserve", job, usd, gpus)))
    monkeypatch.setattr(launcher.ledger, "release", lambda job, actual=None: calls.append(("release", job, actual)))
    monkeypatch.setattr(launcher, "run_modal", lambda jobs: (calls.append(("run", len(jobs))), run_modal(jobs))[1])
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    p1, p2 = write_donor_files(tmp_path)
    argv = ["--tag", "t", "--ckpts", "smoke", "--shards", "2", "--reserve-usd", "7", "--efforts", "low",
            "--run1-prompts", str(p1), "--run2-prompts", str(p2)]
    return launcher, calls, argv


def two_shard_outputs(tmp_path):
    prompts = [{**p, "messages": [{"role": "user", "content": "x" * (i % 7 + 1)}]} for i, p in enumerate(mh.build_prompt_sets(*write_donor_files(tmp_path)))][:6]
    prompts += [{**p, "set": "r2_val"} for p in prompts]
    out = fake_shard_output(tmp_path, prompts)
    return [out, {**out, "shard_id": 1}]


def test_launcher_main_reserves_then_runs_then_releases_with_estimated_cost(tmp_path, monkeypatch, capsys):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, lambda jobs: two_shard_outputs(tmp_path))
    assert launcher.main(argv) == 0
    from v3 import ledger
    expect = ledger.est_cost(mh.GPU, 1, 2 * (12.0 + launcher.CONTAINER_OVERHEAD_S) / 3600)     # two shards, 12 s each + overhead
    assert [c[0] for c in calls] == ["reserve", "run", "release"]
    assert calls[0] == ("reserve", "hf-projection-t", 7.0, 2) and calls[1] == ("run", 2) and calls[2] == ("release", "hf-projection-t", expect)
    record = json.loads((tmp_path / "records" / "projection_t.json").read_text())
    assert record["est_cost_usd"] == expect and record["flags"] == [] and set(record["checkpoints"]) == {"bf16_base", "r1_intact", "r1_refusal_ablated"}
    assert "PLACEHOLDER" not in capsys.readouterr().out and "PLACEHOLDER" not in json.dumps(record)


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_launcher_main_releases_the_reservation_when_the_run_dies(tmp_path, monkeypatch, exc):
    def dies(jobs):
        raise exc
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, dies)
    clock = iter([1000.0, 2800.0])                                                              # t0, then the end of a 30 minute run
    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(time=lambda: next(clock)))
    with pytest.raises(type(exc)):
        launcher.main(argv)
    from v3 import ledger
    assert [c[0] for c in calls] == ["reserve", "run", "release"]                               # released exactly once, then re-raised
    assert calls[2] == ("release", "hf-projection-t", ledger.est_cost(mh.GPU, 1, 2 * 1800.0 / 3600))   # no shard result: wall time x shards
    assert not (tmp_path / "records").exists()


def test_launcher_main_already_reserved_and_dry_run_and_bad_efforts(tmp_path, monkeypatch):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, lambda jobs: two_shard_outputs(tmp_path))
    assert launcher.main(argv + ["--already-reserved", "--job", "mine"]) == 0
    assert [c[0] for c in calls] == ["run", "release"] and calls[1][1] == "mine"                 # no second reservation
    calls.clear()
    assert launcher.main(argv + ["--dry-run"]) == 0 and calls == []                              # dry run touches neither Modal nor ledger
    with pytest.raises(ValueError):
        launcher.main(argv + ["--efforts", "low,bogus"])
    assert calls == []                                                                           # rejected before any reservation


def test_sanity_flags_detect_each_failure_mode(tmp_path):
    launcher = load_launcher()
    outs = two_shard_outputs(tmp_path)   # NB: both fake shards hold the same checkpoint names; the later shard's records win
    args = launcher.argparse.Namespace(tag="t", efforts="low", shards=2, layer=2, batch_size=1, no_crosscheck=False)
    prompts = mh.build_prompt_sets(*write_donor_files(tmp_path))
    assert launcher.build_record(args, prompts, outs, 1.0, 0.0)["flags"] == []

    def flags_after(mutate):
        o = json.loads(json.dumps(outs))
        mutate(o)
        return launcher.build_record(args, prompts, o, 1.0, 0.0)["flags"]

    off = {s: {"min_cos": 0.9, "mean_rel_l2": 0.2, "max_rel_l2": 0.3} for s in mh.SITES}
    assert any("disagrees with HF" in f for f in flags_after(lambda o: o[1]["ckpts"]["bf16_base"].update(swap_vs_hf_from_pretrained=off)))
    assert any("rendered prompts differ" in f for f in flags_after(lambda o: o[1]["ckpts"]["r1_intact"]["checks"]["low"].update(rendered_sha256="0" * 64)))
    assert any("different r1 vector" in f for f in flags_after(lambda o: o[1]["vectors"]["files"]["r1"].update(sha256="f" * 64)))
    assert any("FAILED" in f for f in flags_after(lambda o: o[0].update(hf_reference={"error": "OSError: x"})))

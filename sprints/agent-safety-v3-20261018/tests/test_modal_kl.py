"""CPU tests for v3.modal_kl and scripts/run_kl.py: KL math vs scipy, benign-row selection, sequence encoding, the Modal job
body on toy models (weights, tokenizer, volumes replaced), and the launcher (ledger / Modal faked). All text is placeholder.

The one HF-backed test (tiny random Qwen3_5 model, checks the float32-head structure assumptions against the real
transformers classes) is skipped where transformers is not installed (the sprint .venv has none)."""
import importlib.util
import json
import pathlib
import random
import sys
import types
import zlib

import numpy as np
import pytest
import torch
from scipy.stats import entropy
from torch import nn

from v3 import ledger
from v3 import modal_hf as mh
from v3 import modal_kl as mk

ROOT = pathlib.Path(__file__).resolve().parents[1]
V = 256


# ------------------------------------------------------------------------------------------ toy tokenizer / model
class ToyTok:
    """One token per character (id = ord % 250 + 1); the chat template is '|'.join(contents) + '<think>\\n'."""

    def __init__(self, suffix="<think>\n"):
        self.suffix = suffix

    def apply_chat_template(self, messages, **kw):
        return "|".join(m["content"] for m in messages) + self.suffix

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}


class MergeTok(ToyTok):
    """Like ToyTok but a double newline is ONE token (id 255): re-merges the prompt/completion boundary."""

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "\n\n":
                ids.append(255)
                i += 2
            else:
                ids.append(ord(text[i]) % 250 + 1)
                i += 1
        return {"input_ids": ids}


class ToyBackbone(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.mix = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids)
        mean = h.cumsum(1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        return types.SimpleNamespace(last_hidden_state=self.norm(h + self.mix(mean)))


class ToyCausal(nn.Module):
    """HF-shaped causal LM: .model(input_ids, attention_mask, use_cache).last_hidden_state, .lm_head, logits_to_keep."""

    def __init__(self, seed=0, d=16, dtype=torch.float32, head_scale=4.0, own_logit_scale=1.0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = ToyBackbone(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        with torch.no_grad():
            self.lm_head.weight.mul_(head_scale)
        self.own_logit_scale = own_logit_scale
        self.calls = 0
        self.to(dtype)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        self.calls += 1
        h = self.model(input_ids).last_hidden_state
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=self.lm_head(h) * self.own_logit_scale)


def softmax64(logits):
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64), -1).numpy()


# ------------------------------------------------------------------------------------------ KL math
@pytest.mark.parametrize("chunk", [1, 3, 8, 1000])
def test_kl_rows_matches_scipy_entropy(chunk):
    rng = np.random.default_rng(0)
    a = torch.tensor(rng.normal(size=(7, 60)) * 3, dtype=torch.float32)
    b = torch.tensor(rng.normal(size=(7, 60)) * 3, dtype=torch.float32)
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    got = mk.kl_rows(la, lb, chunk=chunk)
    want = entropy(softmax64(a), softmax64(b), axis=1)             # scipy: sum pk log(pk / qk), nats
    assert got.shape == (7,) and got.dtype == np.float64
    assert np.allclose(got, want, atol=2e-6, rtol=1e-5)
    # direction: KL(P_ref || P_x) is the first argument of scipy's entropy, and is not symmetric
    assert np.allclose(mk.kl_rows(lb, la), entropy(softmax64(b), softmax64(a), axis=1), atol=2e-6, rtol=1e-5)
    assert not np.allclose(mk.kl_rows(la, lb), mk.kl_rows(lb, la), rtol=1e-3)


def test_kl_rows_large_dynamic_range_and_vocab_size():
    rng = np.random.default_rng(1)
    a = torch.tensor(rng.normal(size=(4, 5000)) * 12, dtype=torch.float32)       # |logit| up to ~50: tiny probabilities
    b = a + torch.tensor(rng.normal(size=(4, 5000)) * 0.3, dtype=torch.float32)
    got = mk.kl_rows(torch.log_softmax(a, -1), torch.log_softmax(b, -1), chunk=3)
    assert np.allclose(got, entropy(softmax64(a), softmax64(b), axis=1), atol=2e-5, rtol=1e-4)


def test_kl_rows_properties():
    rng = np.random.default_rng(2)
    la = torch.log_softmax(torch.tensor(rng.normal(size=(5, 40)), dtype=torch.float32), -1)
    lb = torch.log_softmax(torch.tensor(rng.normal(size=(5, 40)), dtype=torch.float32), -1)
    assert np.all(mk.kl_rows(la, la) == 0.0)                                      # identical log-probs: exactly 0
    assert np.all(mk.kl_rows(la, lb) > 0)
    assert np.allclose(mk.kl_rows(la, lb, chunk=2), mk.kl_rows(la, lb, chunk=5))   # chunking invariance
    with pytest.raises(ValueError):
        mk.kl_rows(la, lb[:, :39])
    with pytest.raises(ValueError):
        mk.kl_rows(la[0], lb[0])
    with pytest.raises(ValueError):
        mk.kl_rows(la, lb, chunk=0)


def test_kl_rows_zero_probability_reference_entries_contribute_nothing():
    p = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    q = np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])
    with np.errstate(divide="ignore"):
        got = mk.kl_rows(torch.tensor(np.log(p)), torch.tensor(np.log(q)))
    assert np.allclose(got, entropy(p, q, axis=1)) and np.isfinite(got).all()


def test_kl_rows_reference_on_another_device_kind_dtype():
    """The host reference is float32 and the checkpoint rows may be float32 or float64: dtype handling only."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(3, 20)), rng.normal(size=(3, 20))
    la, lb = torch.log_softmax(torch.tensor(a), -1), torch.log_softmax(torch.tensor(b), -1)
    assert np.allclose(mk.kl_rows(la.float(), lb), mk.kl_rows(la, lb), atol=1e-6)


# ------------------------------------------------------------------------------------------ last_logprobs
def test_last_logprobs_shapes_rows_and_both_paths_agree_in_float32():
    model = ToyCausal(seed=1)
    ids = [int(x) for x in np.random.default_rng(0).integers(1, 250, size=40)]
    fp32 = mk.last_logprobs(model, ids, 33, True, "cpu")
    own = mk.last_logprobs(model, ids, 33, False, "cpu")
    assert fp32.shape == own.shape == (33, V) and fp32.dtype == torch.float32
    assert torch.allclose(fp32.exp().sum(-1), torch.ones(33), atol=1e-5)
    assert torch.allclose(fp32, own, atol=1e-5)
    with torch.no_grad():
        full = model(torch.tensor([ids])).logits[0]
    assert torch.allclose(fp32, torch.log_softmax(full[-33:], -1), atol=1e-5)    # the LAST 33 positions of the sequence
    one = mk.last_logprobs(model, ids, 1, True, "cpu")
    assert torch.allclose(one[0], torch.log_softmax(full[-1], -1), atol=1e-5)


def test_fp32_head_is_closer_to_float64_truth_than_bf16_logits():
    model = ToyCausal(seed=2, dtype=torch.bfloat16, head_scale=12.0)
    ids = [int(x) for x in np.random.default_rng(1).integers(1, 250, size=48)]
    fp32 = mk.last_logprobs(model, ids, 33, True, "cpu")
    own = mk.last_logprobs(model, ids, 33, False, "cpu")
    with torch.no_grad():
        h = model.model(torch.tensor([ids])).last_hidden_state[0, -33:].double()
        truth = torch.log_softmax(h @ model.lm_head.weight.double().T, -1).float()
    assert (fp32 - truth).abs().max() < 1e-4                  # float32 head: only float32 summation error
    assert (own - truth).abs().max() > 20 * (fp32 - truth).abs().max()   # bf16 logits are visibly coarser
    assert float(mk.kl_rows(own, fp32).max()) < mk.HEAD_CHECK_MAX_KL      # ... but still within the job's head-check bound


def test_fp32_linear_chunking_matches_unchunked_to_float32_precision():
    rng = np.random.default_rng(4)
    h = torch.tensor(rng.normal(size=(5, 16)), dtype=torch.bfloat16)
    w = torch.tensor(rng.normal(size=(100, 16)), dtype=torch.bfloat16)
    assert torch.allclose(mk.fp32_linear(h, w, chunk=7), mk.fp32_linear(h, w, chunk=1000), atol=1e-5)
    assert torch.allclose(mk.fp32_linear(h, w), h.float() @ w.float().T)


# ------------------------------------------------------------------------------------------ transcripts and selection
def fake_row(tid, set_name="C", variant="C", consequence="harmless", effort=None, n_tokens=120, prompt=None, completion=None,
             finish="stop", cot="cot_on", kind="single"):
    prompt = prompt if prompt is not None else f"[BENIGN_TEXT_PLACEHOLDER_{tid}]<think>\n"
    return {"trial_id": tid, "exp_id": "phase1", "kind": kind, "cot_mode": cot, "reasoning_effort": effort, "max_new_tokens": 16384,
            "messages": [{"role": "user", "content": f"[BENIGN_TEXT_PLACEHOLDER_{tid}]"}], "full_prompt": prompt,
            "raw_output": completion if completion is not None else f"[COMPLETION_PLACEHOLDER_{tid}]" + "z" * n_tokens,
            "generated_tokens": n_tokens, "finish_reason": finish, "think_closed": True, "arm": "intact",
            "meta": {"set_name": set_name, "variant": variant, "consequence": consequence}}


def fake_transcript(n_c=60, n_a0=60, extra=True):
    rows = []
    for i in range(n_c):
        rows.append(fake_row(f"C_h_{i:03d}", "C", "C"))
    for i in range(n_a0):
        rows.append(fake_row(f"A0_h_{i:03d}", "A", "A0"))
    if extra:    # everything the selection must ignore
        for i in range(20):
            rows.append(fake_row(f"A1_h_{i:03d}", "A", "A1"))
            rows.append(fake_row(f"A2_h_{i:03d}", "A", "A2"))
            rows.append(fake_row(f"C_x_{i:03d}", "C", "C", consequence="harmful"))
            rows.append(fake_row(f"A0_x_{i:03d}", "A", "A0", consequence="harmful"))
            rows.append(fake_row(f"C_low_{i:03d}", "C", "C", effort="low"))
            rows.append(fake_row(f"A0_low_{i:03d}", "A", "A0", effort="low"))
            rows.append(fake_row(f"B_{i:03d}", "B", "B", consequence="harmful"))
    return rows


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_row_label_filters():
    assert mk.row_label(fake_row("a", "C", "C")) == "C" and mk.row_label(fake_row("a", "A", "A0")) == "A0"
    for bad in (fake_row("a", "A", "A1"), fake_row("a", "A", "A2"), fake_row("a", "B", "B"), fake_row("a", consequence="harmful"),
                fake_row("a", effort="low"), fake_row("a", cot="cot_off"), fake_row("a", kind="episode")):
        assert mk.row_label(bad) is None


def test_selection_is_balanced_distinct_deterministic_and_order_independent(tmp_path):
    rows = fake_transcript()
    r1 = write_jsonl(tmp_path / "r1.jsonl", rows)
    cands, stats = mk.load_benign_candidates(r1)
    assert len(cands) == 120 and stats["n_r1_rows"] == 120
    sel = mk.select_benign_rows(cands)
    assert len(sel) == 48 and [c["label"] for c in sel].count("C") == 24 and [c["label"] for c in sel].count("A0") == 24
    assert all(("_h_" in c["id"]) for c in sel)                                      # harmless, default effort, C / A0 only
    assert len({c["full_prompt"] for c in sel}) == 48 and len({c["id"] for c in sel}) == 48
    for seed in range(5):
        shuffled = list(cands)
        random.Random(seed).shuffle(shuffled)
        assert [c["id"] for c in mk.select_benign_rows(shuffled)] == [c["id"] for c in sel]
    # extra rows that the filter ignores do not change the selection
    more = write_jsonl(tmp_path / "r1b.jsonl", fake_transcript(extra=False))
    assert [c["id"] for c in mk.select_benign_rows(mk.load_benign_candidates(more)[0])] == [c["id"] for c in sel]
    assert [c["id"] for c in mk.select_benign_rows(cands, salt="other")] != [c["id"] for c in sel]       # the salt matters
    small = mk.select_benign_rows(cands, n=10)
    assert len(small) == 10 and sum(c["label"] == "C" for c in small) == 5


def test_selection_dedupes_by_prompt_and_tops_up_when_a_set_is_short(tmp_path):
    rows = fake_transcript(n_c=10, n_a0=60, extra=False)
    rows += [fake_row("C_dup", "C", "C", prompt=rows[0]["full_prompt"])]            # same full_prompt as an existing row
    cands, _ = mk.load_benign_candidates(write_jsonl(tmp_path / "r1.jsonl", rows))
    sel = mk.select_benign_rows(cands)
    assert len(sel) == 48 and [c["label"] for c in sel].count("C") == 10 and [c["label"] for c in sel].count("A0") == 38
    assert len({c["full_prompt"] for c in sel}) == 48
    few = mk.select_benign_rows(cands[:5])
    assert len(few) == 5


def test_candidates_with_r2_transcript(tmp_path):
    r1 = fake_transcript(n_c=4, n_a0=4, extra=False)
    r2 = [dict(r, raw_output="[R2_COMPLETION_PLACEHOLDER]" + "q" * 100, arm="intact") for r in r1]
    r2 = [r for r in r2 if r["trial_id"] != "C_h_000"]                              # missing in r2 -> dropped
    r2[0]["generated_tokens"] = 5                                                    # too short in r2 -> dropped
    cands, stats = mk.load_benign_candidates(write_jsonl(tmp_path / "a.jsonl", r1), write_jsonl(tmp_path / "b.jsonl", r2))
    assert stats["n_missing_r2"] == 1 and stats["n_short_or_unfinished_r2"] == 1 and len(cands) == 6
    assert all(set(c["completions"]) == {"r1", "r2"} and c["completions"]["r1"] != c["completions"]["r2"] for c in cands)
    r2[3]["full_prompt"] = "[OTHER_PLACEHOLDER]<think>\n"
    with pytest.raises(ValueError, match=r2[3]["trial_id"]) as err:
        mk.load_benign_candidates(write_jsonl(tmp_path / "a.jsonl", r1), write_jsonl(tmp_path / "b.jsonl", r2))
    assert "PLACEHOLDER" not in str(err.value)
    dup = write_jsonl(tmp_path / "d.jsonl", r1 + r1[:1])
    with pytest.raises(ValueError, match="duplicate"):
        mk.load_benign_candidates(dup)


def test_short_or_unfinished_completions_are_not_eligible(tmp_path):
    rows = [fake_row("C_h_000", n_tokens=mk.MIN_GEN_TOKENS - 1), fake_row("C_h_001", finish="length"), fake_row("C_h_002")]
    cands, stats = mk.load_benign_candidates(write_jsonl(tmp_path / "r1.jsonl", rows))
    assert [c["id"] for c in cands] == ["C_h_002"] and stats["n_short_or_unfinished_r1"] == 2


REAL_R1 = ROOT / "raw_transcripts/phase1_dim_r1/intact.jsonl"
REAL_R2 = ROOT / "raw_transcripts/phase1_dim_r2/intact.jsonl"


@pytest.mark.skipif(not (REAL_R1.exists() and REAL_R2.exists()), reason="local transcripts not on disk")
def test_real_transcripts_selection_structure_only():
    """Structural assertions only (no text is ever formatted into a message)."""
    cands, stats = mk.load_benign_candidates(REAL_R1, REAL_R2)
    sel = mk.select_benign_rows(cands)
    assert len(sel) == 48 and sum(c["label"] == "C" for c in sel) == 24 and sum(c["label"] == "A0" for c in sel) == 24
    assert len({c["full_prompt"] for c in sel}) == 48 and stats["n_missing_r2"] == 0
    assert all(c["full_prompt"].endswith("<think>\n") and "</think>" not in c["full_prompt"] for c in sel)
    assert all(c["id"].endswith("__eff_default") for c in sel)
    assert [c["id"] for c in sel] == [c["id"] for c in mk.select_benign_rows(list(reversed(cands)))]


# ------------------------------------------------------------------------------------------ encode_sequence
def test_encode_sequence_prefix_truncation_and_boundary_fallback():
    tok = ToyTok()
    prompt, comp = "[BENIGN_TEXT_PLACEHOLDER]<think>\n", "abc" * 30
    enc = mk.encode_sequence(tok, prompt, comp)
    p = tok(prompt)["input_ids"]
    assert enc["n_prompt"] == len(p) and len(enc["ids"]) == len(p) + 32 and enc["ids"][:len(p)] == p
    assert enc["ids"][len(p):] == tok(comp)["input_ids"][:32] and enc["joint_tokenization"] is True
    # boundary merge: the prompt's final "\n" would merge with the completion's leading "\n"
    merged = mk.encode_sequence(MergeTok(), prompt, "\n" + comp)
    assert merged["joint_tokenization"] is False
    assert merged["ids"][:merged["n_prompt"]] == MergeTok()(prompt)["input_ids"]
    assert merged["ids"][merged["n_prompt"]:] == MergeTok()("\n" + comp)["input_ids"][:32]
    with pytest.raises(ValueError, match="need 32") as err:
        mk.encode_sequence(tok, prompt, "short")
    assert "short" not in str(err.value) and "PLACEHOLDER" not in str(err.value)
    assert mk.ids_sha256([enc]) == mk.ids_sha256([dict(enc)]) and mk.ids_sha256([enc]) != mk.ids_sha256([merged])


# ------------------------------------------------------------------------------------------ aggregation
def test_variants_and_summaries_on_known_matrices():
    n = 4
    r1 = np.zeros((n, 33))
    r1[:, 0] = [0.1, 0.2, 0.3, 0.4]
    r1[:, 1:] = np.arange(n)[:, None] * 0.01
    r2 = r1 * 2
    v = mk.variants_from_matrices({"r1": r1, "r2": r2}, "r1_pca_101")
    assert v["kl_prompt_last"] == pytest.approx([0.1, 0.2, 0.3, 0.4]) and v["kl_teacher_forced"] == pytest.approx([0, 0.01, 0.02, 0.03])
    assert v["kl_teacher_forced_own"] == pytest.approx(v["kl_teacher_forced"])                    # r1 family: own = r1 trajectory
    v2 = mk.variants_from_matrices({"r1": r1, "r2": r2}, "r2_pca_101")
    assert v2["kl_teacher_forced"] == pytest.approx(v["kl_teacher_forced"]) and v2["kl_teacher_forced_own"] == pytest.approx([0, 0.02, 0.04, 0.06])
    assert mk.variants_from_matrices({"r1": r1}, "r2_intact")["kl_teacher_forced_own"] == pytest.approx(v["kl_teacher_forced"])   # no r2 pass
    st = mk.summarize_pair({"r1": r1, "r2": r2}, "r1_x")
    assert st["kl_prompt_last"]["mean"] == pytest.approx(0.25) and st["kl_prompt_last"]["median"] == pytest.approx(0.25)
    assert st["kl_prompt_last"]["max"] == pytest.approx(0.4) and st["kl_prompt_last"]["n"] == 4
    assert st["kl_prompt_last"]["gate_mean_lt_0.10"] is False and st["kl_teacher_forced"]["gate_mean_lt_0.10"] is True
    assert st["pos0_traj_consistency_max_abs_diff"] == pytest.approx(0.4) and len(st["per_position_mean_r1"]) == 33
    assert mk.agg([1, 2, 10])["median"] == 2 and mk.agg([1, 2, 10])["max"] == 10


def test_primary_ref_and_checkpoint_order():
    assert mk.primary_ref("r1_pca_101") == "r1_intact" and mk.primary_ref("r2_unrelated") == "r2_intact"
    assert mk.primary_ref("r1_intact") == "r1_intact" and mk.primary_ref("bf16_base") is None
    names = [c["name"] for c in mh.select_checkpoints("all")]
    assert len(names) == 13
    order = [c["name"] for c in mk.order_checkpoints(list(reversed(mh.select_checkpoints("all"))))]
    assert order[:3] == ["bf16_base", "r1_intact", "r2_intact"] and sorted(order) == sorted(names)


def test_modal_app_wiring():
    assert mk.app.name == "agent-safety-v3-20261018-kl"
    assert mk.OUT_DIR == "/artifacts/agent-safety-v3-20261018/kl" and mk.GPU == "H200" and mk.GATE == 0.10
    assert callable(mk.kl_checkpoints.remote) and callable(mk.kl_checkpoints.local)
    assert mk.N_PROMPTS == 48 and mk.N_GEN == 32 and mk.N_POS == 33


# ------------------------------------------------------------------------------------------ the Modal job body, on CPU
class _Vol:
    def reload(self):
        pass

    def commit(self):
        self.commits = getattr(self, "commits", 0) + 1


SEEDS = {"bf16_base": 10, "r1_intact": 11, "r2_intact": 12, "r1_refusal_ablated": 13, "r2_pca_101": 14, "r1_pca_101": 15}


def seed_of(path: str) -> int:
    """Seed of a toy checkpoint: by name for the tests that use names as paths, by crc32 for real /weights paths."""
    return SEEDS.get(path, zlib.crc32(path.encode()) % 1000)


def toy_rows(n=6, two_trajectories=True):
    tok = ToyTok()
    rows = []
    for i in range(n):
        messages = [{"role": "user", "content": f"[BENIGN_TEXT_PLACEHOLDER_{i}]" + "x" * i}]
        comp = {"r1": "".join(chr(97 + (7 * i + 3 * j) % 26) for j in range(60))}
        if two_trajectories:
            comp["r2"] = "".join(chr(97 + (5 * i + 11 * j) % 26) for j in range(60))
        rows.append({"id": f"toy_{i}", "label": "C" if i % 2 == 0 else "A0", "messages": messages,
                     "full_prompt": tok.apply_chat_template(messages), "completions": comp})
    return rows


def install_toys(monkeypatch, tmp_path, tokenizers=None, models=None):
    """Replace weights, tokenizers, volumes and HF classes of the job body by toys. `tokenizers`: path -> tokenizer."""
    tokenizers = tokenizers or {}
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "stub"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return tokenizers.get(path, ToyTok())

    fake_tf.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    vol = _Vol()
    built = []

    def build(base_dir, device):
        m = ToyCausal(seed=99, **(models or {}).get("build", {}))
        built.append(m)
        return m

    monkeypatch.setattr(mk, "art_vol", vol)
    monkeypatch.setattr(mk, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mh, "build_model", build)
    monkeypatch.setattr(mh, "check_config_matches_base", lambda *a, **k: None)

    def load(model, path):
        src = ToyCausal(seed=seed_of(path), **(models or {}).get("src", {}))
        model.load_state_dict(src.state_dict())
        return {"n_params": 0}

    monkeypatch.setattr(mh, "load_checkpoint_into_model", load)
    return vol, built


def run_toy_job(monkeypatch, tmp_path, names, rows=None, **kw):
    vol, built = install_toys(monkeypatch, tmp_path, **kw)
    rows = rows if rows is not None else toy_rows()
    job = {"tag": "t", "ckpts": [{"name": n, "path": n} for n in names], "rows": rows, "n_gen": 32, "repeat_prompts": 2, "device": "cpu"}
    return mk.kl_checkpoints.local(job), rows, vol


def manual_logprobs(seed, ids, dtype=torch.float32):
    model = ToyCausal(seed=seed, dtype=dtype)
    with torch.no_grad():
        return torch.log_softmax(model(torch.tensor([ids])).logits[0, -33:].double(), -1)


def test_job_body_on_cpu_kl_matches_independent_scipy(monkeypatch, tmp_path):
    names = ["r2_pca_101", "r1_refusal_ablated", "bf16_base", "r1_intact", "r2_intact", "r1_pca_101"]
    res, rows, vol = run_toy_job(monkeypatch, tmp_path, names)
    assert list(res["ckpts"]) == ["bf16_base", "r1_intact", "r2_intact", "r2_pca_101", "r1_refusal_ablated", "r1_pca_101"]   # references first
    assert res["config"]["trajectories"] == ["r1", "r2"] and res["config"]["n_prompts"] == 6 and vol.commits >= 4
    assert res["checks"]["head_check"]["fp32_head_used"] is True and res["checks"]["repeat_check"]["max_kl_between_two_runs"] == 0.0
    tok = ToyTok()
    for name, rec in res["ckpts"].items():
        assert rec["diag"]["render_matches"] == 6 and rec["diag"]["tokenization_same_as_first_ckpt"] and rec["n_nonfinite_kl"] == 0
        for traj in ("r1", "r2"):
            for ref, mats in ((r, rec["kl"][r]) for r in rec["kl"]):
                assert np.asarray(mats[traj]).shape == (6, 33) and np.asarray(mats[traj]).min() > -1e-9
    # self KL is exactly zero for the three references
    for ref in mk.REFERENCES:
        for traj in ("r1", "r2"):
            assert np.all(np.asarray(res["ckpts"][ref]["kl"][ref][traj]) == 0.0)
    # every cell recomputed independently with the toy models and scipy.stats.entropy (never touching modal_kl internals)
    for name in ("r1_refusal_ablated", "r2_pca_101", "bf16_base"):
        for ref in res["ckpts"][name]["kl"]:
            for traj in ("r1", "r2"):
                for i in (0, 3, 5):
                    ids = mk.encode_sequence(tok, rows[i]["full_prompt"], rows[i]["completions"][traj])["ids"]
                    p, q = manual_logprobs(SEEDS[ref], ids).exp().numpy(), manual_logprobs(SEEDS[name], ids).exp().numpy()
                    want = entropy(p, q, axis=1)
                    assert np.allclose(res["ckpts"][name]["kl"][ref][traj][i], want, atol=1e-5, rtol=1e-4), (name, ref, traj, i)
    # the two trajectories are different token streams (so the r2 pass really used the r2 completions), same position 0
    r1m, r2m = (np.asarray(res["ckpts"]["r2_pca_101"]["kl"]["r2_intact"][t]) for t in ("r1", "r2"))
    assert not np.allclose(r1m[:, 1:], r2m[:, 1:]) and np.allclose(r1m[:, 0], r2m[:, 0], atol=1e-5)
    # outputs: JSON on the 'volume' per checkpoint, serializable result, no prompt/completion text anywhere
    files = sorted(p.name for p in (tmp_path / "out" / "t").glob("*.json"))
    assert files == sorted(f"{n}.json" for n in names)
    blob = json.dumps(res) + "".join(p.read_text() for p in (tmp_path / "out" / "t").glob("*.json"))
    assert "PLACEHOLDER" not in blob
    on_disk = json.loads((tmp_path / "out" / "t" / "r1_refusal_ablated.json").read_text())
    assert on_disk["kl"] == json.loads(json.dumps(res["ckpts"]["r1_refusal_ablated"]["kl"]))
    summ = mk.build_summary(res["ckpts"])
    assert summ["r1_refusal_ablated"]["primary_ref"] == "r1_intact" and set(summ["r1_refusal_ablated"]["vs"]) == set(mk.REFERENCES)
    assert summ["bf16_base"]["primary_ref"] is None and summ["r2_pca_101"]["vs"]["r2_intact"]["kl_teacher_forced_own"]["mean"] > 0


def test_job_body_without_r2_trajectory_and_subset_of_references(monkeypatch, tmp_path):
    rows = toy_rows(4, two_trajectories=False)
    res, _, _ = run_toy_job(monkeypatch, tmp_path, ["r1_refusal_ablated", "r1_intact"], rows=rows)
    assert res["config"]["trajectories"] == ["r1"] and res["config"]["references"] == ["r1_intact"]
    rec = res["ckpts"]["r1_refusal_ablated"]
    assert set(rec["kl"]) == {"r1_intact"} and set(rec["kl"]["r1_intact"]) == {"r1"}
    assert np.all(np.asarray(res["ckpts"]["r1_intact"]["kl"]["r1_intact"]["r1"]) == 0.0)
    st = mk.summarize_pair(rec["kl"]["r1_intact"], "r1_refusal_ablated")
    assert "pos0_traj_consistency_max_abs_diff" not in st and st["kl_teacher_forced_own"]["mean"] == st["kl_teacher_forced"]["mean"]


def test_job_uses_first_checkpoints_ids_when_a_tokenizer_differs(monkeypatch, tmp_path):
    rows = toy_rows(3)
    rows[1]["completions"]["r1"] = "\nab\n\ncd" + rows[1]["completions"]["r1"]   # leading newline re-merges the boundary; "\n\n" is one MergeTok token
    res, _, _ = run_toy_job(monkeypatch, tmp_path, ["bf16_base", "r1_intact", "r1_refusal_ablated"], rows=rows,
                            tokenizers={"r1_refusal_ablated": MergeTok()})
    ok, bad = res["ckpts"]["r1_intact"]["diag"], res["ckpts"]["r1_refusal_ablated"]["diag"]
    assert ok["tokenization_same_as_first_ckpt"] is True and bad["tokenization_same_as_first_ckpt"] is False
    assert bad["tokens_sha256"]["r1"] != ok["tokens_sha256"]["r1"]
    assert bad["n_joint_tokenization"]["r1"] == 2 and ok["n_joint_tokenization"]["r1"] == 3            # its own tokenizer's boundary merges
    tok = ToyTok()                                                                # KL was computed on the first checkpoint's ids
    ids = mk.encode_sequence(tok, rows[1]["full_prompt"], rows[1]["completions"]["r1"])["ids"]
    want = entropy(manual_logprobs(11, ids).exp().numpy(), manual_logprobs(13, ids).exp().numpy(), axis=1)
    assert np.allclose(res["ckpts"]["r1_refusal_ablated"]["kl"]["r1_intact"]["r1"][1], want, atol=1e-5, rtol=1e-4)


def test_job_falls_back_to_own_logits_when_the_float32_head_check_fails(monkeypatch, tmp_path):
    """A model whose own logits are not backbone x lm_head (here: rescaled) fails the head check; the job then uses its logits."""
    res, rows, _ = run_toy_job(monkeypatch, tmp_path, ["bf16_base", "r1_intact"], models={"src": {"own_logit_scale": 0.5}, "build": {"own_logit_scale": 0.5}})
    chk = res["checks"]["head_check"]
    assert chk["fp32_head_used"] is False and chk["kl_ownlogits_vs_fp32head_max"] > mk.HEAD_CHECK_MAX_KL
    ids = mk.encode_sequence(ToyTok(), rows[2]["full_prompt"], rows[2]["completions"]["r1"])["ids"]
    own = lambda seed: torch.log_softmax(ToyCausal(seed=seed, own_logit_scale=0.5)(torch.tensor([ids])).logits[0, -33:].detach().double(), -1).exp().numpy()
    assert np.allclose(res["ckpts"]["r1_intact"]["kl"]["bf16_base"]["r1"][2], entropy(own(10), own(11), axis=1), atol=1e-5, rtol=1e-4)


def test_job_rejects_bad_prompts_and_missing_r1(monkeypatch, tmp_path):
    rows = toy_rows(2)
    rows[0]["full_prompt"] = "[BENIGN_TEXT_PLACEHOLDER]</think>\n<think>\n"
    with pytest.raises(ValueError, match="toy_0") as err:
        run_toy_job(monkeypatch, tmp_path, ["bf16_base"], rows=rows)
    assert "PLACEHOLDER" not in str(err.value)
    rows = toy_rows(2)
    rows[1]["full_prompt"] = "[BENIGN_TEXT_PLACEHOLDER]"
    with pytest.raises(ValueError, match="toy_1"):
        run_toy_job(monkeypatch, tmp_path, ["bf16_base"], rows=rows)
    with pytest.raises(ValueError, match="r1 trajectory"):
        run_toy_job(monkeypatch, tmp_path, ["bf16_base"], rows=[])


def test_job_raises_on_non_finite_logprobs(monkeypatch, tmp_path):
    class NanModel(ToyCausal):
        def get_output_embeddings(self):
            w = self.lm_head
            bad = nn.Linear(w.in_features, w.out_features, bias=False)
            bad.weight.data = torch.full_like(w.weight, float("nan"))
            return bad

    install_toys(monkeypatch, tmp_path)
    monkeypatch.setattr(mh, "build_model", lambda base_dir, device: NanModel(seed=1))
    monkeypatch.setattr(mh, "load_checkpoint_into_model", lambda model, path: {})
    job = {"tag": "t", "ckpts": [{"name": "bf16_base", "path": "bf16_base"}], "rows": toy_rows(2), "device": "cpu"}
    with pytest.raises((RuntimeError, ValueError)):
        mk.kl_checkpoints.local(job)


# ------------------------------------------------------------------------------------------ launcher
def load_launcher():
    path = ROOT / "scripts" / "run_kl.py"
    spec = importlib.util.spec_from_file_location("run_kl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy_transcripts(tmp_path):
    r1 = fake_transcript(n_c=30, n_a0=30)
    r2 = [dict(r, raw_output=r["raw_output"][::-1]) for r in r1]
    return write_jsonl(tmp_path / "r1.jsonl", r1), write_jsonl(tmp_path / "r2.jsonl", r2)


def launcher_with_fakes(tmp_path, monkeypatch, run_modal):
    launcher = load_launcher()
    calls = []
    monkeypatch.setattr(launcher.ledger, "reserve", lambda job, usd, gpus=1: calls.append(("reserve", job, usd, gpus)))
    monkeypatch.setattr(launcher.ledger, "release", lambda job, actual=None: calls.append(("release", job, actual)))
    monkeypatch.setattr(launcher, "run_modal", lambda job: (calls.append(("run", [c["name"] for c in job["ckpts"]], len(job["rows"]))), run_modal(job))[1])
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    r1, r2 = toy_transcripts(tmp_path)
    argv = ["--tag", "t", "--ckpts", "r1_intact,r1_refusal_ablated,bf16_base,r2_intact", "--r1-transcript", str(r1), "--r2-transcript", str(r2)]
    return launcher, calls, argv


def toy_run_modal(monkeypatch, tmp_path):
    def run(job):
        # the job body on toys; ids of the toy tokenizer differ from the fake transcripts' 'full_prompt' only by rendering
        # (messages are rendered by ToyTok), so make the rows consistent first
        tok = ToyTok()
        rows = [dict(r, full_prompt=tok.apply_chat_template(r["messages"]), completions={t: c + "z" * 40 for t, c in r["completions"].items()}) for r in job["rows"]]
        install_toys(monkeypatch, tmp_path)
        return mk.kl_checkpoints.local({**job, "rows": rows, "device": "cpu"})
    return run


def test_launcher_rejects_forbidden_prompts_before_any_reservation(tmp_path, monkeypatch):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, lambda job: None)
    rows = fake_transcript(n_c=30, n_a0=30, extra=False)
    rows[0]["full_prompt"] = "[BENIGN_TEXT_PLACEHOLDER] </think>\n<think>\n"
    rows[1]["full_prompt"] = "[BENIGN_TEXT_PLACEHOLDER] DECLARED_TOOL_NAME<think>\n"
    rows[2]["full_prompt"] = "[BENIGN_TEXT_PLACEHOLDER_no_suffix]"
    write_jsonl(tmp_path / "r1.jsonl", rows)
    write_jsonl(tmp_path / "r2.jsonl", rows)
    with pytest.raises(ValueError, match="3 prompts") as err:
        launcher.main(argv + ["--n-prompts", "60"])
    assert "PLACEHOLDER" not in str(err.value) and calls == []


def test_launcher_dry_run_prints_counts_only(tmp_path, monkeypatch, capsys):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, lambda job: None)
    assert launcher.main(argv + ["--dry-run"]) == 0 and calls == []
    out = capsys.readouterr().out
    assert "prompts=48" in out and "'C': 24" in out and "'A0': 24" in out and "PLACEHOLDER" not in out
    assert "trajectories=r1,r2" in out


def test_launcher_main_reserves_runs_releases_writes_record_and_prints_aggregates(tmp_path, monkeypatch, capsys):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, None)
    monkeypatch.setattr(launcher, "run_modal", lambda job: (calls.append(("run", [c["name"] for c in job["ckpts"]], len(job["rows"]))), toy_run_modal(monkeypatch, tmp_path)(job))[1])
    assert launcher.main(argv + ["--n-prompts", "12"]) == 0
    assert [c[0] for c in calls] == ["reserve", "run", "release"]
    assert calls[0] == ("reserve", "kl-t", 10.0, 1) and calls[1][2] == 12
    record = json.loads((tmp_path / "records" / "kl_t.json").read_text())
    expect = ledger.est_cost(mk.GPU, 1, (record["elapsed_s"] + launcher.CONTAINER_OVERHEAD_S) / 3600)
    assert calls[2] == ("release", "kl-t", expect) and record["est_cost_usd"] == expect
    assert record["schema"] == "kl_v1" and record["gate"] == 0.10 and record["prompt_set"]["n"] == 12
    assert record["flags"] == ["only 12 benign prompts (target 48)"]
    assert set(record["checkpoints"]) == {"bf16_base", "r1_intact", "r2_intact", "r1_refusal_ablated"}
    ab = record["checkpoints"]["r1_refusal_ablated"]
    assert ab["primary_ref"] == "r1_intact" and set(ab["vs"]) == {"bf16_base", "r1_intact", "r2_intact"}
    st = ab["vs"]["r1_intact"]
    for variant in ("kl_prompt_last", "kl_teacher_forced", "kl_teacher_forced_own"):
        arr = np.asarray(st[variant]["per_prompt"])
        assert len(arr) == 12 and st[variant]["mean"] == pytest.approx(arr.mean()) and st[variant]["median"] == pytest.approx(np.median(arr))
        assert st[variant]["max"] == pytest.approx(arr.max())
    mats = np.asarray(record["kl_position_matrix"]["r1_refusal_ablated"]["r1_intact"]["r1"])
    assert mats.shape == (12, 33) and st["kl_prompt_last"]["per_prompt"] == pytest.approx(mats[:, 0].tolist())
    assert st["kl_teacher_forced"]["per_prompt"] == pytest.approx(mats[:, 1:].mean(1).tolist())
    assert record["prompt_set"]["transcripts"]["r1"]["sha256"] == launcher.sha256_file(tmp_path / "r1.jsonl")
    blob = json.dumps(record)
    shown = capsys.readouterr().out
    assert "PLACEHOLDER" not in blob and "PLACEHOLDER" not in shown
    assert "released kl-t" in shown and "r1_refusal_ablated" in shown and "kl_prompt_last" in shown and ("PASS" in shown or "FAIL" in shown)
    for fid in record["prompt_set"]["ids"]:
        assert fid not in shown                                                        # not even trial ids are printed


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_launcher_releases_the_reservation_when_the_run_dies(tmp_path, monkeypatch, exc):
    def dies(job):
        raise exc
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, dies)
    clock = iter([1000.0, 2800.0])                                                     # t0, then the end of a 30 minute run
    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(time=lambda: next(clock)))
    with pytest.raises(type(exc)):
        launcher.main(argv)
    assert [c[0] for c in calls] == ["reserve", "run", "release"]
    assert calls[2] == ("release", "kl-t", ledger.est_cost(mk.GPU, 1, 1800.0 / 3600)) and not (tmp_path / "records").exists()


def test_launcher_already_reserved_and_job_name(tmp_path, monkeypatch):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, None)
    monkeypatch.setattr(launcher, "run_modal", lambda job: (calls.append(("run", 0, 0)), toy_run_modal(monkeypatch, tmp_path)(job))[1])
    assert launcher.main(argv + ["--already-reserved", "--job", "kl-full13", "--n-prompts", "6"]) == 0
    assert [c[0] for c in calls] == ["run", "release"] and calls[1][1] == "kl-full13"


def test_reserve_with_wait_retries_on_concurrency_only(monkeypatch):
    launcher = load_launcher()
    seq = iter([RuntimeError("concurrency: 16+1 GPUs > 16"), RuntimeError("concurrency: 16+1 GPUs > 16"), None])
    sleeps = []

    def fake_reserve(job, usd, gpus=1):
        r = next(seq)
        if r is not None:
            raise r

    monkeypatch.setattr(launcher.ledger, "reserve", fake_reserve)
    t = [0.0]
    launcher.reserve_with_wait("j", 10, 1, wait_min=30, poll_s=60, sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)), clock=lambda: t[0])
    assert sleeps == [60, 60]
    monkeypatch.setattr(launcher.ledger, "reserve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("budget: projected $460 > soft stop")))
    with pytest.raises(RuntimeError, match="budget"):
        launcher.reserve_with_wait("j", 10, 1, wait_min=30, sleep=lambda s: None)
    monkeypatch.setattr(launcher.ledger, "reserve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("concurrency: 16+1 GPUs > 16")))
    t[0] = 0.0
    with pytest.raises(RuntimeError, match="concurrency"):
        launcher.reserve_with_wait("j", 10, 1, wait_min=3, poll_s=60, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert t[0] <= 3 * 60 + 60                                                           # gave up at the deadline


def test_sanity_flags_detect_each_failure_mode(tmp_path, monkeypatch):
    launcher, calls, argv = launcher_with_fakes(tmp_path, monkeypatch, None)
    args = launcher.argparse.Namespace(tag="t", r1_transcript=str(tmp_path / "r1.jsonl"), r2_transcript=str(tmp_path / "r2.jsonl"), no_r2_trajectory=False)
    rows, stats = launcher.build_rows(launcher.argparse.Namespace(n_prompts=6, no_r2_trajectory=False, r1_transcript=args.r1_transcript, r2_transcript=args.r2_transcript))
    tok = ToyTok()
    rows = [dict(r, full_prompt=tok.apply_chat_template(r["messages"]), completions={t: c + "z" * 40 for t, c in r["completions"].items()}) for r in rows]
    install_toys(monkeypatch, tmp_path)
    out = mk.kl_checkpoints.local({"tag": "t", "rows": rows, "device": "cpu", "ckpts": [{"name": n, "path": n} for n in ("bf16_base", "r1_intact", "r2_intact", "r1_refusal_ablated")]})
    base = json.loads(json.dumps(out))
    assert launcher.build_record(args, rows, stats, base, 1.0, 0.0)["flags"] == ["only 6 benign prompts (target 48)"]    # nothing else

    def flags_after(mutate):
        o = json.loads(json.dumps(base))
        mutate(o)
        return launcher.build_record(args, rows, stats, o, 1.0, 0.0)["flags"]

    assert any("own logits used" in f for f in flags_after(lambda o: o["checks"]["head_check"].update(fp32_head_used=False)))
    assert any("two identical forward passes" in f for f in flags_after(lambda o: o["checks"]["repeat_check"].update(max_kl_between_two_runs=0.01)))
    assert any("matched the transcript prompt" in f for f in flags_after(lambda o: o["ckpts"]["r1_intact"]["diag"].update(render_matches=3)))
    assert any("different ids" in f for f in flags_after(lambda o: o["ckpts"]["r1_intact"]["diag"].update(tokenization_same_as_first_ckpt=False)))
    assert any("non-finite" in f for f in flags_after(lambda o: o["ckpts"]["r1_intact"].update(n_nonfinite_kl=2)))
    assert any("against itself" in f for f in flags_after(lambda o: o["ckpts"]["r1_intact"]["kl"]["r1_intact"]["r1"][0].__setitem__(1, 0.5)))
    assert any("position-0" in f for f in flags_after(lambda o: o["ckpts"]["r1_refusal_ablated"]["kl"]["r1_intact"]["r2"][0].__setitem__(0, 9.0)))


# ------------------------------------------------------------------------------------------ real HF architecture (optional)
def test_tiny_qwen3_5_float32_head_matches_logits_to_keep():
    """Random tiny Qwen3_5ForCausalLM: the structure assumptions of last_logprobs (model.model(...).last_hidden_state is
    post final norm, get_output_embeddings().weight is the head, logits_to_keep keeps the last rows) hold on the real classes."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model("qwen3_5")
    tc = cfg.text_config
    tc.num_hidden_layers, tc.layer_types = 4, ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.hidden_size, tc.intermediate_size, tc.vocab_size = 128, 256, 300
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 32
    for k, v in (("linear_num_key_heads", 2), ("linear_num_value_heads", 4), ("linear_key_head_dim", 16), ("linear_value_head_dim", 16)):
        setattr(tc, k, v)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    ids = [int(x) for x in np.random.default_rng(0).integers(1, 290, size=45)]
    fp32 = mk.last_logprobs(model, ids, 33, True, "cpu")
    own = mk.last_logprobs(model, ids, 33, False, "cpu")
    assert fp32.shape == own.shape == (33, 300)
    assert float(mk.kl_rows(own, fp32).max()) < mk.HEAD_CHECK_MAX_KL and (fp32 - own).abs().max() < 0.2
    with torch.no_grad():
        full = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    assert torch.allclose(own, torch.log_softmax(full[-33:], -1), atol=1e-3)      # logits_to_keep keeps the LAST rows


def test_job_body_on_real_tiny_qwen3_5_with_real_loader_and_fp8_checkpoints(monkeypatch, tmp_path):
    """The whole job body on real transformers classes: tiny random Qwen3_5, synthetic block-FP8 checkpoints in the real
    naming, the real dequantizing swap loader; KL recomputed from HF's own logits with scipy."""
    pytest.importorskip("transformers")
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model("qwen3_5")
    tc = cfg.text_config
    tc.num_hidden_layers, tc.layer_types = 4, ["linear_attention"] * 3 + ["full_attention"]
    tc.hidden_size, tc.intermediate_size, tc.vocab_size = 128, 256, 300
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 32
    for k, v in (("linear_num_key_heads", 2), ("linear_num_value_heads", 4), ("linear_key_head_dim", 16), ("linear_value_head_dim", 16)):
        setattr(tc, k, v)
    base = tmp_path / "base"
    base.mkdir()
    cfg.save_pretrained(base)

    def quantize(w, block=128):
        m, n = w.shape
        scale = w.float().reshape(m // block, block, n // block, block).abs().amax(dim=(1, 3)) / 448.0
        tiles = w.float().reshape(m // block, block, n // block, block) / scale[:, None, :, None]
        return tiles.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(m, n).contiguous(), scale

    def make_ckpt(name, seed):
        torch.manual_seed(seed)
        src = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        d = tmp_path / name
        d.mkdir()
        tensors = {}
        for n_, p in src.state_dict().items():
            key = n_.replace("model.", "model.language_model.", 1) if n_.startswith("model.") else n_
            if n_.endswith("proj.weight") and p.ndim == 2 and p.shape[0] % 128 == 0 and p.shape[1] % 128 == 0:
                tensors[key], tensors[key + "_scale_inv"] = quantize(p)
            else:
                tensors[key] = p.contiguous()
        save_file(tensors, str(d / "all.safetensors"))
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "all.safetensors" for k in tensors}}))
        (d / "config.json").write_text((base / "config.json").read_text())
        return str(d)

    names = ["bf16_base", "r1_intact", "r2_intact", "r1_refusal_ablated"]
    paths = {n: make_ckpt(n, 20 + i) for i, n in enumerate(names)}
    monkeypatch.setattr(mk, "art_vol", _Vol())
    monkeypatch.setattr(mk, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mk, "BASE_DIR", str(base))
    orig_check = mh.check_config_matches_base
    monkeypatch.setattr(mh, "check_config_matches_base", lambda p: orig_check(p, str(base)))
    stub_tok = types.SimpleNamespace(from_pretrained=lambda p: ToyTok())            # the toy character tokenizer (vocab 300 > its ids)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=stub_tok, AutoConfig=AutoConfig, AutoModelForCausalLM=AutoModelForCausalLM, __version__=transformers.__version__))
    rows = toy_rows(3)
    res = mk.kl_checkpoints.local({"tag": "e2e", "ckpts": [{"name": n, "path": paths[n]} for n in names], "rows": rows, "device": "cpu", "repeat_prompts": 1})
    assert res["checks"]["head_check"]["fp32_head_used"] is True and res["checks"]["repeat_check"]["max_kl_between_two_runs"] == 0.0
    assert res["ckpts"]["r1_intact"]["diag"]["load"]["n_fp8_dequantized"] == 14
    ids = mk.encode_sequence(ToyTok(), rows[2]["full_prompt"], rows[2]["completions"]["r2"])["ids"]

    def hf_probs(name):
        m = mh.build_model(str(base), "cpu")
        mh.load_checkpoint_into_model(m, paths[name])
        with torch.no_grad():
            return torch.softmax(m(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -33:].double(), -1).numpy()

    want = entropy(hf_probs("r2_intact"), hf_probs("r1_refusal_ablated"), axis=1)
    got = np.asarray(res["ckpts"]["r1_refusal_ablated"]["kl"]["r2_intact"]["r2"][2])
    assert want.mean() > 1e-3 and np.allclose(got, want, atol=2e-4)       # the toy checkpoints really differ; bf16-logit rounding bound
    assert np.all(np.asarray(res["ckpts"]["r2_intact"]["kl"]["r2_intact"]["r2"]) == 0.0)

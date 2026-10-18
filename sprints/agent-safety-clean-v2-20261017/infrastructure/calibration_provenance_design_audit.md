# Supplemental calibration provenance and format-v2 admission

The separate `scripts/audit_calibration_provenance.py` independently reconstructs
the complete calibration request journal, raw completion sequence, source-trial
metadata, sandbox observations, assistant/user histories, and full rendered
prompt bytes. It checks exact immutable Intact checkpoint paths, saved
trajectories, pilot Markdown, terminal worker counts, and the worker's loaded
FP8 inventory evidence. The existing numerical calibration receipt remains
unchanged. Missing capture or terminal artifacts produce PENDING, never PASS.

`scripts/modal_capture_calibration_provenance.py` prepares one bounded CPU-only
call: 2 CPUs, 4 GiB RAM, 1800-second execution plus 300-second startup limit,
one container, no retries, no GPU and no inference. It verifies all eighteen
actual checkpoint shards against the original immutable Intact manifest and
captures tokenizer/configuration bytes to a new, exclusively claimed directory.
It also preserves and recovers the two immutable execution files that the old
calibration watcher did not download: `sandbox_inputs/` and `group_calls/`.
The same-namespace original BF16 tokenizer files are compared when available.

After central resource admission, the intended successful-pilot attempt is:

```text
.venv/bin/modal run scripts/modal_capture_calibration_provenance.py --attempt author_v1_c2 --capture-id cpu_v1
```

A completed capture can be recovered with the same command plus
`--recover-only`, which makes no new remote function call. Interrupted or failed
captures are preserved and need a new capture ID. Never pay to capture the
archived c1 failed pilot. The capture may run after c2 H5 succeeds while its full
sweep continues; the independent audit requires the full terminal evidence.

The manifests created by the original Intact pipeline record weight-shard
hashes but no historical tokenizer-file hashes. The running worker omitted
generation-time tokenizer digests and input token IDs. Accordingly this audit
claims post-launch, checkpoint-bound exact prompt consistency; it does not
claim a generation-time tokenizer attestation. Input token IDs, if recorded,
are independently checked; their absence alone does not fail H4. Every raw
prompt still must replay byte-for-byte from the captured template and history.

The fresh format-v2 admission uses the same forty authored scenarios and five
fresh pilots, with the renderer's explicit complete tool-call envelope. The
controller's optional `--config-path` binds the immutable configuration path
through child audits, cache identity, snapshots and the Modal command. New
preparation uses `--output stimuli/calibration_v1_format_v2.json`; its exclusive
publication preserves the original v1 file. Attempt `author_v1_c2` receives a
fresh cache. Attempt c1's original cache/source/config snapshots are retained;
control-plane recovery cannot authorize a changed scientific payload.

Validation: nine admission/recovery/config-version tests, eight synthetic
provenance tests and fifteen aggregate-readiness tests pass. The provenance
tests reject jointly forged request/raw prompts, forged feedback, changed
checkpoint paths, missing execution evidence, corrupted actual shards,
overwritten captures and changed tokenizer bytes. No paid calls, launches,
scientific receipt writes, ledger changes or GPU actions were performed by
this implementation task.

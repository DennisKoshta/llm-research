# Calibration admission recovery audit

The 2026-10-18 04:34:35 UTC admission attempt failed before reserving or launching calibration: `ValueError: Resource ledger exceeds authorized cap`. The ledger records its single-worker amendment from four to six at 04:32:38 UTC, with the global ten-GPU and one-TP8 limits unchanged. The on-disk launcher was modified at 04:32:38 UTC. The authorization cache and execution snapshot were created afterward, at 04:34:44 UTC, and bind the updated launcher SHA-256 `c8ca1809f6992654d39c1240e26f4c6020e5386c7df3cf6940836e08d8e95de5`.

The supervisor imported `launch` at module startup, before waiting for authoring. Its earlier log already shows admission polling before 04:04 UTC. Its audit subprocess therefore checked updated on-disk code while the long-lived parent called its older imported function. The current launcher permits six single workers; the old imported version rejected a ledger limit of six. PID 370055 was absent when inspected, and the ledger contained no `phase1_intact_calibration_author_v1_c1` reservation. This is a control-plane version mismatch, not a failed scientific gate.

The repaired supervisor compiles the current central launcher directly from source and delegates launch admission to a bounded fresh child retaining the attempt admission lock. It does not change the central launch authority or resource ledger. Launch configuration includes all immutable prerequisite hashes, checked by the central authority under its own lock.

The explicit `--recover-control-plane` mode preserves the original authorization cache, scientific authorization, source snapshot, resource configuration, model inputs, attempt, and job ID. It verifies the original controller/launcher snapshots against their cached hashes, permits updates only to those two admission files, and writes their new versions and old/new hashes into a separate immutable `control_plane_recovery/<hash>/` subdirectory when the parent runs recovery. Every other original bound file must still match. Existing job reservations trigger observer recovery rather than another paid launch. Unknown launch windows and duplicate budget reservations remain fail-closed.

Read-only verification succeeded for all 35 bound prerequisites, and the rebuilt execution configuration was identical to the original frozen configuration. The current central resource audit requires an $11 minimum compute reservation; the original configuration still requests $12. No actual recovery record, cache, ledger, research receipt, process, or GPU job was changed during this audit.

Six focused regression tests passed. They cover old imported limits versus current compiled limits, exact scientific-cache preservation, rejection of scientific/payload/old-snapshot mutations, current-source hash binding at launch, inherited lock propagation, capacity wait classification, and existing-reservation deduplication.

Parent recovery command, from this sprint directory:

```bash
.venv/bin/python scripts/admit_parallel_calibration.py --author-attempt v1 --attempt author_v1_c1 --recover-control-plane
```

Use the parent's normal detached-process/log wrapper. The previously reported PID is already gone; no termination was needed. The specification's Phase 1 calibration rule, Intact-only design, independent prerequisite audit, 4096-token decoding, five fresh pilots, and all scientific thresholds remain unchanged. This command does not authorize Phase 2.

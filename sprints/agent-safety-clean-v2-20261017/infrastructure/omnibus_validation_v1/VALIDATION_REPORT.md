# Synthetic omnibus CPU validation — complete

This report contains **synthetic infrastructure validation only**. It is not an
empirical result, research receipt, complete experiment, or Phase 3 authorization.

The actual PyMC NUTS model ran successfully on a constructed separation fixture.
Both ITT and strict fits passed every registered convergence threshold. A fresh
direct rerun reproduced all retained posterior and sampler-statistic arrays and
all summaries exactly. A second full rerun through the isolated subprocess bridge
also reproduced them exactly, while the main research Python process imported no
NumPy, SciPy, PyMC, PyTensor, or ArviZ modules.

| Denominator | n | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Depth hits | Min BFMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| itt | 2880 | 1.002764 | 6228.64 | 3831.66 | 0 | 0 | 0.933813 |
| strict | 2872 | 1.002994 | 6583.25 | 3783.74 | 0 | 0 | 0.905113 |

Each full fit used four sequential chains, 2,000 warmup and 2,000 retained draws,
float64, target acceptance 0.95 and maximum tree depth 12. The synthetic fixture
contains 80 constructed events and eight constructed malformed episodes; these
numbers describe test data only. All 139 diagnostic components, 37 coefficients,
ten three-way interactions and 72 predictive cells were checked. The separate
all-zero 100-warmup/100-draw smoke produced finite, nondegenerate intervals but
did not meet production convergence thresholds; its failure remains visible.

The isolated `.omnibus-venv` uses the complete compatible lock
`requirements_lock_v2.txt`; `pip check` passes. All 12 scientific-module tests and
13 subprocess tests pass. CPU affinity was restricted to cores 0 and 1, numerical
threads to one, and chains ran sequentially. Resource snapshots show one active
thread and approximately 250–280 MB resident memory. No GPUs, Modal calls,
empirical outcomes, research receipts, or existing environment mutations were used.

Two engineering incompatibilities were retained and resolved:

1. ArviZ 0.21.0 could not import with initially resolved Matplotlib 3.11.2. Pinning
   Matplotlib 3.10.1 in the isolated environment fixed it. The original lock,
   failed import, corrected lock and complete package metadata are preserved.
2. The first environment inventory exposes additional setuptools bundled
   metadata during threadpool initialization. The wrapper now performs the same
   explicit preflight as the validated driver, then requires exact equality of
   the full initialized environment (49 metadata entries versus 38 initially).
   The rejected wrapper attempt and its exact source snapshot are retained.
   No fingerprint field was omitted and no missing-package fallback was added.

The scientific source, its tests and frozen `exp11_omnibus_plan_v1.json` remain
byte-for-byte unchanged. No superseding scientific plan was necessary. The new
stdlib-only bridge is `scripts/exp11_omnibus_subprocess.py`:
`recompute(episodes, destination, audit_directory=<fresh directory>, timeout_seconds=3600)`.
It returns the original scientific replay dictionary unchanged. Every call must
use a new audit directory; it never skips sampling based on a cache. Timeouts,
nonzero exits, changed inputs/sources and mismatched output hashes fail closed.
Execution metadata stays separate from scientific posterior inputs and receipts.

After the actual bridge run completed, its exact source was preserved. A final
file-integrity-only hardening then bound both posterior manifests and all 22
declared array files before and after replay. Mutation regressions reject changes
to an array or a consistent replacement of both array and manifest. The hardened
binding function was checked against all 30 real saved artifact/source bindings;
the full sampler was not rerun for this file-check-only change. Both exact bridge
source versions and their validation scope are recorded in
`post_hardening_validation.json`; the scientific model source remains unchanged.

For future CPU fitting, perform `scientific.environment()` once before
`scientific.fit(...)`, matching the validated driver. Final verification must use
the isolated wrapper with independently audited raw-derived episodes. Exact
replay is established only in this recorded environment; cross-platform bitwise
equality is not claimed. PyTensor could not link C-BLAS, so the unchanged default
compiled CPU backend was slower; no alternate backend or sampler was substituted.

`VALIDATION_REPORT.json` contains exact timings and source hashes.
`ARTIFACT_MANIFEST.json` hashes every retained validation artifact. The README
provides the standalone direct replay command.

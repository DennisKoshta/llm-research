# Synthetic CPU integration validation

Every input, posterior array, component JSON, diagnostic, and predictive check in this directory is **synthetic infrastructure validation only**. None is an empirical study result, an experiment receipt, a Phase 1/2/3 completion claim, or authorization for another phase. The fixture uses invented scenario identities and constructed outcomes. In particular, generic `component.json` files produced by the production analysis API here remain synthetic.

The prospective protocol is `synthetic_validation_protocol.json`. The original frozen statistical module, tests, and analysis plan are preserved with `frozen_v1_` names. Sampling uses the original statistical module without scientific changes. Full registered settings mean four sequential chains, 2,000 warmup plus 2,000 retained draws per chain, for each of ITT and strict denominators. The separate all-zero smoke uses only 100 warmup/100 retained draws and does not establish production convergence.

The isolated environment is `.omnibus-venv` at the sprint root. `requirements_lock_v2.txt` is the compatible full dependency lock. The original lock and failed import are retained: ArviZ 0.21.0 is incompatible with the initially resolved Matplotlib 3.11.2; pinning Matplotlib 3.10.1 repairs the environment without changing model priors, sampler settings, source, or prospective analysis plan.

CPU affinity is limited to cores 0 and 1; numerical thread pools and sequential sampling use one thread. `environment.json`, `authorization_and_limits.json`, and `sampler_resource_observation.json` record the environment and resource checks. No GPUs or Modal calls are used. PyTensor reported unavailable C-BLAS linkage; its default compiled CPU backend is retained, with no backend substitution.

Exact sampler replay is a separate required validation step, not implied by a successful initial fit. Diagnostic failures remain visible and do not trigger an automatic model refit or weakened threshold. Consult the final validation report for completed steps and limitations.

To repeat the complete saved synthetic posterior audit in the validated local
environment, run from the sprint root:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
MPLCONFIGDIR="$PWD/.omnibus-cache/matplotlib" \
NUMBA_CACHE_DIR="$PWD/.omnibus-cache/numba" \
PYTENSOR_FLAGS="base_compiledir=$PWD/.omnibus-cache/pytensor,floatX=float64" \
TMPDIR="$PWD/.omnibus-cache/tmp" \
taskset -c 0,1 .omnibus-venv/bin/python -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"scripts"); import exp11_omnibus_analysis as m; p=Path("infrastructure/omnibus_validation_v1"); r=m.recompute(json.loads((p/"synthetic_separated_episodes.json").read_text()),p/"separated_full_v1",replay_sampler=True); print(r["raw_to_sampler_replay_verified"],r["saved_chain_summary_recomputed"])'
```

This reruns both full samplers and reads existing artifacts without overwriting
them. The validation driver itself writes exclusive operation reports, so its
already-completed `fit`/`replay` report paths must not be reused for another run.
Exact replay is claimed only for the recorded Python/package/compiler/platform
environment, not across hardware or software changes.

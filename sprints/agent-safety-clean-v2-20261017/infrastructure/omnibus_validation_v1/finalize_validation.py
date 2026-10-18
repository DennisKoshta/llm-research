"""Finalize exclusively synthetic CPU integration evidence after actual success."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]


def read(name):
    return json.loads((FOLDER/name).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(name, value):
    with (FOLDER/name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')


smoke = read('all_zero_smoke/synthetic_smoke_summary.json')
fit = read('separated_full_v1_fit_validation.json')
replay = read('separated_full_v1_replay_validation.json')
bridge = read('subprocess_integration_validation.json')
execution = read('subprocess_replay_v2/execution.json')
component = read('separated_full_v1/component.json')
initialized = read('subprocess_replay_v2/environment_initialization.json')
hardening = read('post_hardening_validation.json')
assert smoke['all_retained_values_finite'] and not smoke['production_fit_complete']
assert component['omnibus_diagnostics_passed'] and not component['complete_experiment_receipt']
assert all(v['status'] == 'CONVERGED' for v in fit['summary'].values())
assert replay['summary'] == {'raw_to_sampler_replay_verified': True, 'saved_chain_summary_recomputed': True}
assert bridge['exact_sampler_replay_via_isolated_subprocess'] and bridge['exact_summary_recomputed']
assert bridge['caller_imported_no_numerical_libraries'] and not bridge['phase3_authorized']
assert execution['status'] == 'SUCCESS' and execution['fresh_sampler_replay'] and not execution['cached_skip']
assert initialized['initialized'] == component['invocation']['environment']
assert (FOLDER/'pip_check.log').read_text().strip() == 'No broken requirements found.'
for name, count in [('unit_tests.log', 12), ('subprocess_hardened_unit_tests.log', 13)]:
    text = (FOLDER/name).read_text()
    assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
source_hashes = {}
for name, subdir in [('exp11_omnibus_analysis.py', 'scripts'),
                     ('test_exp11_omnibus_analysis.py', 'tests'),
                     ('exp11_omnibus_plan_v1.json', 'infrastructure')]:
    source = ROOT/subdir/name
    assert source.read_bytes() == (FOLDER/('frozen_v1_'+name)).read_bytes()
    source_hashes[str(source.relative_to(ROOT))] = sha(source)
for name in ['scripts/exp11_omnibus_subprocess.py', 'tests/test_exp11_omnibus_subprocess.py']:
    source_hashes[name] = sha(ROOT/name)
value = {'schema': 'synthetic_omnibus_integration_validation_v1',
         'scope': 'SYNTHETIC INFRASTRUCTURE VALIDATION ONLY; NO EMPIRICAL STUDY RESULTS',
         'status': 'COMPLETE', 'completed_utc': datetime.now(timezone.utc).isoformat(),
         'all_zero_smoke_finite': True, 'all_zero_smoke_production_convergence_claim': False,
         'full_separated_fit': fit['summary'], 'direct_full_sampler_replay_exact': True,
         'isolated_subprocess_full_sampler_replay_exact': True,
         'main_environment_imported_no_numerical_libraries': True,
         'main_environment_package_versions': bridge['caller_versions'],
         'unit_tests_passed': 25, 'pip_check_passed': True,
         'posterior_file_binding_hardening': hardening,
         'full_dependency_lock': 'requirements_lock_v2.txt',
         'full_dependency_lock_sha256': sha(FOLDER/'requirements_lock_v2.txt'),
         'initial_and_corrected_failure_evidence_preserved': True,
         'scientific_model_sampler_source_tests_and_v1_plan_unchanged': True,
         'source_sha256': source_hashes,
         'fit_elapsed_seconds': fit['elapsed_seconds'],
         'direct_replay_elapsed_seconds': replay['elapsed_seconds'],
         'subprocess_replay_elapsed_seconds': execution['elapsed_seconds'],
         'empirical_study_completed': False, 'complete_experiment_receipt': False,
         'phase3_authorized': False}
write('VALIDATION_REPORT.json', value)
rows = []
for name, entry in fit['summary'].items():
    rows.append(f"| {name} | {entry['n']} | {entry['rhat_max']:.6f} | {entry['bulk_ess_min']:.2f} | {entry['tail_ess_min']:.2f} | {entry['divergences']} | {entry['depth_hits']} | {min(entry['bfmi']):.6f} |")
markdown = '''# Synthetic omnibus CPU validation — complete

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
'''+ '\n'.join(rows)+'''

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
'''
with (FOLDER/'VALIDATION_REPORT.md').open('x') as handle:
    handle.write(markdown)
write('ARTIFACT_MANIFEST.json', {str(path.relative_to(FOLDER)): sha(path)
      for path in sorted(FOLDER.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_MANIFEST.json'})
print('SYNTHETIC integration complete; exact direct and isolated-subprocess sampler replay verified.')

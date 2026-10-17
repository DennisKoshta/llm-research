"""Recompute B's frozen analysis into a new directory; no model calls or refits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', choices=['pilot', 'main', 'repaired_pilot_v2', 'repaired_main_v2'], required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    source = HERE / 'runs' / args.run; destination = args.output.resolve()
    assert not destination.exists(), 'Use a new directory; never replace an existing replay'
    manifest = json.loads((HERE / 'executions/cost_gradient_v1/manifest.json').read_text())
    assert sha(HERE / 'analysis.py') == manifest['files']['analysis.py']
    destination.mkdir(parents=True)
    import analysis
    original_save = analysis.save
    allowed = {'scored.json', 'analysis.json', 'audit.json'}

    def isolated_save(path, value):
        assert path.parent == source and path.name in allowed
        return original_save(destination / path.name, value)

    analysis.save = isolated_save
    analysis.analyze(source)
    comparisons = {}
    for name in sorted(allowed):
        actual = json.loads((destination / name).read_text()); expected = json.loads((source / name).read_text())
        if name == 'analysis.json':
            actual.pop('created_utc'); expected.pop('created_utc')
        comparisons[name] = actual == expected
        assert comparisons[name], name
    if (source / 'secondary-consequences-v2.json').exists():
        freeze = json.loads((HERE / 'secondary-consequence-freeze-v2.json').read_text())
        for path, expected_sha in freeze['files'].items():
            assert sha(Path(path)) == expected_sha
        subprocess.run([sys.executable, str(HERE / 'secondary_consequence_analysis_v2.py'), str(destination)], check=True)
        name = 'secondary-consequences-v2.json'
        comparisons[name] = json.loads((source / name).read_text()) == json.loads((destination / name).read_text())
        assert comparisons[name]
    receipt = {'passed': all(comparisons.values()), 'run': args.run,
               'exact_json_comparisons_excluding_creation_timestamp': comparisons,
               'analysis_source_sha256': sha(HERE / 'analysis.py'), 'replay_script_sha256': sha(Path(__file__)),
               'original_inputs_and_outputs_preserved': True, 'inference_calls': 0,
               'source_analysis_sha256': sha(source / 'analysis.json'), 'source_directory': str(source)}
    (destination / 'reproduction-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

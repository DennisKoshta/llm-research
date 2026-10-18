"""Stdlib-only bridge to immutable, isolated CPU NUTS replay.

The caller supplies independently audited episodes. This bridge does not certify
that upstream audit and never authorizes empirical sampling or a study phase.
Every call requires a fresh audit_directory; no cached or summary-only success.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp11_omnibus_subprocess_request_v1'
MAX_TIMEOUT_SECONDS = 3600


class ReplayProcessError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Omnibus replay path escapes sprint')
    return path


def relative(path):
    return str(inside(path).relative_to(ROOT.resolve()))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def environment():
    env = dict(os.environ)
    # Do not inherit alternate Python module/installation injection paths.
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONHASHSEED='0', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=str(ROOT/'.omnibus-cache/matplotlib'),
               NUMBA_CACHE_DIR=str(ROOT/'.omnibus-cache/numba'),
               PYTENSOR_FLAGS='base_compiledir='+str(ROOT/'.omnibus-cache/pytensor')+',floatX=float64',
               TMPDIR=str(ROOT/'.omnibus-cache/tmp'))
    return env


def bindings(destination):
    paths = [ROOT/'scripts/exp11_omnibus_analysis.py',
             ROOT/'scripts/exp11_omnibus_subprocess.py',
             ROOT/'infrastructure/exp11_omnibus_plan_v1.json',
             ROOT/'infrastructure/omnibus_validation_v1/requirements_lock_v2.txt',
             destination/'invocation.json', destination/'component.json']
    return {relative(path): sha(path) for path in paths}


def validate_request(request_path):
    request_path = inside(request_path)
    require(request_path.name == 'request.json', 'Unexpected replay request name')
    folder = request_path.parent
    request = json.loads(request_path.read_text())
    require(request['schema'] == SCHEMA and request['replay_sampler'] is True,
            'Only complete sampler replay is supported')
    require(request['audit_directory'] == relative(folder), 'Replay audit directory differs')
    destination = inside(ROOT/request['destination'])
    require(request['bindings'] == bindings(destination), 'Replay source or posterior binding changed')
    input_path = inside(ROOT/request['episodes_path'])
    require(input_path == folder/'audited_episodes.json' and sha(input_path) == request['episodes_sha256'],
            'Audited episode bytes changed')
    require(request['cpu_affinity'] == [0, 1] and request['numerical_threads'] == 1,
            'Replay resource settings changed')
    require(type(request['timeout_seconds']) is int and 1 <= request['timeout_seconds'] <= MAX_TIMEOUT_SECONDS,
            'Replay timeout outside bound')
    return request, destination, input_path


def child(request_path):
    request, destination, input_path = validate_request(request_path)
    require(set(os.sched_getaffinity(0)) <= {0, 1}, 'Replay CPU affinity exceeds two CPUs')
    for key, expected in environment().items():
        if key in ('PYTHONHASHSEED', 'PYTHONNOUSERSITE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR',
                   'PYTENSOR_FLAGS', 'TMPDIR'):
            require(os.environ.get(key) == expected, 'Replay environment differs: '+key)
    require(Path(sys.prefix).resolve() == (ROOT/'.omnibus-venv').resolve(), 'Wrong isolated interpreter')
    import exp11_omnibus_analysis as scientific
    result = scientific.recompute(json.loads(input_path.read_text()), destination, replay_sampler=True)
    require(result['raw_to_sampler_replay_verified'] is True and result['saved_chain_summary_recomputed'] is True,
            'Scientific replay did not complete')
    require(result['phase3_authorized'] is False, 'Omnibus component cannot authorize Phase3')
    # Fail on any input/source/posterior-manifest drift during a long replay.
    validate_request(request_path)
    output = Path(request_path).parent/'result.json'
    write(output, result)
    write(output.with_name('result_binding.json'), {'request_sha256': sha(request_path),
          'result_sha256': sha(output), 'episodes_sha256': request['episodes_sha256'],
          'raw_to_sampler_replay_verified': True})


def recompute(episodes, destination, *, audit_directory, timeout_seconds=3600):
    """Return scientific recompute dict unchanged after a fresh isolated replay.

    Requires a NEW audit_directory. Repeated calls must choose another directory;
    FileExistsError is intentional. No importing numerical libraries in caller.
    """
    require(type(timeout_seconds) is int and 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS,
            'Replay timeout must be 1..3600 seconds')
    require(isinstance(episodes, list), 'Expected independently audited episode list')
    destination = inside(destination); folder = inside(audit_directory)
    require(not folder.is_relative_to(destination), 'Replay evidence cannot be nested inside saved posterior')
    require(not destination.is_relative_to(folder), 'Replay directory cannot contain saved posterior')
    interpreter = ROOT/'.omnibus-venv/bin/python'
    require(interpreter.is_file(), 'Isolated omnibus interpreter is missing')
    source_bindings = bindings(destination)
    require({0, 1} <= set(os.sched_getaffinity(0)), 'Required replay CPU affinity is unavailable')
    folder.mkdir(parents=True, exist_ok=False)
    input_path = folder/'audited_episodes.json'; write(input_path, episodes)
    request = {'schema': SCHEMA, 'destination': relative(destination),
               'audit_directory': relative(folder), 'episodes_path': relative(input_path),
               'episodes_sha256': sha(input_path), 'bindings': source_bindings,
               'replay_sampler': True, 'cpu_affinity': [0, 1], 'numerical_threads': 1,
               'timeout_seconds': timeout_seconds}
    request_path = folder/'request.json'; write(request_path, request)
    command = ['taskset', '-c', '0,1', str(interpreter),
               str(ROOT/'scripts/exp11_omnibus_subprocess.py'), '--request', str(request_path)]
    started = time.monotonic(); process = None
    try:
        with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise ReplayProcessError('Isolated sampler replay exceeded its CPU time bound') from exc
        require(code == 0, 'Isolated sampler replay exited with code '+str(code))
        validate_request(request_path)
        result_path = folder/'result.json'; proof = json.loads((folder/'result_binding.json').read_text())
        require(proof == {'request_sha256': sha(request_path), 'result_sha256': sha(result_path),
                          'episodes_sha256': request['episodes_sha256'], 'raw_to_sampler_replay_verified': True},
                'Isolated replay output provenance differs')
        result = json.loads(result_path.read_text())
        require(result['raw_to_sampler_replay_verified'] is True and result['saved_chain_summary_recomputed'] is True
                and result['phase3_authorized'] is False, 'Isolated replay returned incomplete result')
        write(folder/'execution.json', {'status': 'SUCCESS', 'command': command,
              'elapsed_seconds': time.monotonic()-started, 'returncode': code,
              'request_sha256': sha(request_path), 'result_sha256': sha(result_path),
              'fresh_sampler_replay': True, 'cached_skip': False})
        return result
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        write(folder/'execution.json', {'status': 'FAILED', 'exception_type': type(exc).__name__,
              'message': str(exc), 'elapsed_seconds': time.monotonic()-started,
              'request_sha256': sha(request_path), 'fresh_sampler_replay': True, 'cached_skip': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    child(parser.parse_args().request)

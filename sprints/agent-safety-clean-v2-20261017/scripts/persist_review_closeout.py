"""Immutable review snapshot preparation and explicit, readback-verified upload.

Default CLI execution is local only. --upload explicitly performs the remote
write; it never launches a Modal app or GPU. FAIL_INCOMPLETE remains a truthful
snapshot status and is never converted into research-program completion.
"""
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
VOLUME = 'curiosity-study-artifacts'
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
REQUIRED = (
    'FINAL_RESEARCH_SPECIFICATION.md', 'FINAL_RESULTS_FOR_REVIEW.md',
    'STAGE0_HYGIENE_REPORT.md', 'PROGRESS_AND_GATES.md', 'attempt_registry.json',
    'ENGINEERING_CORRECTIONS.md', 'ENGINEERING_CORRECTIONS_COMPLETED_STATUS.md',
    'verification/verification_status.json',
)
INFRA_PATTERNS = (
    'primary_q4_independent_failure_audit.*', 'calibration_*_independent_provenance.json',
    'calibration_format_v2_correction.json', 'phase1_descriptive_dose*.json',
    'dose_*rule*.json', 'dose_d3_unlaunched_disposition.json', 'dose_tokenizer_metadata_recovery_v1.json',
    'budget_*.json', 'resource_*.json', 'modal_environment_resource_limits.json',
    '*terminal*provenance.json', '*terminal*provenance.md', '*pilot_independent_audit.json',
    '*_watch_status.json', '*_status.json', '*coverage*amendment*.json',
    'calibration_provenance/*/*/capture.json', 'calibration_provenance/*/*/capture_request.json',
    'author_provenance/*/*/capture.json', 'author_provenance/*/*/fixed_prompt_replay.json',
)
EXECUTION_NAMES = {'source_manifest.json', 'resource_config.json', 'frozen_config.json',
                   'frozen_authorization.json', 'authorization_cache.json'}
DENIED_COMPONENT = re.compile(r'(?i)(^\.|(?:^|[_-])(?:secrets?|credentials?|passwords?|private[_-]?keys?|api[_-]?keys?)(?:[._-]|$))')
TOKEN_LITERAL = re.compile(rb'(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{24,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,})(?![A-Za-z0-9])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def safe_path(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    if not isinstance(name, str) or not name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('A sprint-relative path without traversal is required')
    if any(DENIED_COMPONENT.search(part) for part in relative.parts):
        raise ValueError('Hidden or credential-like paths are excluded')
    path = root
    for part in relative.parts:
        path = path/part
        if path.is_symlink():
            raise ValueError('Symlinks are excluded from closeout')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes sprint')
    return path


def allowed(name, include_sources=False):
    path = PurePosixPath(name); parts = path.parts
    if name in REQUIRED:
        return True
    if len(parts) == 2 and parts[0] in {'records', 'verification'} and path.suffix == '.json':
        return True
    if len(parts) == 2 and parts[0] == 'pilots' and path.suffix == '.md':
        return True
    if parts and parts[0] == 'infrastructure' and path.suffix in {'.json', '.md'}:
        # Deep provenance is selected only by exact filename; tokenizer/model/raw
        # payloads and numerical sampler arrays are deliberately not included.
        return len(parts) == 2 or path.name in {'capture.json', 'capture_request.json', 'fixed_prompt_replay.json'}
    if parts and parts[0] == 'executions' and path.name in EXECUTION_NAMES:
        return 'source_files' not in parts
    return include_sources and len(parts) == 2 and parts[0] in {'scripts', 'tests'} and path.suffix == '.py'


def select_paths(root, extra_paths=(), include_sources=False):
    root = Path(root).resolve(); names = set(REQUIRED)
    for pattern in ('records/*.json', 'verification/*.json', 'pilots/*.md'):
        names.update(str(path.relative_to(root)) for path in root.glob(pattern))
    for pattern in INFRA_PATTERNS:
        names.update(str(path.relative_to(root)) for path in (root/'infrastructure').glob(pattern))
    for basename in EXECUTION_NAMES:
        names.update(str(path.relative_to(root)) for path in (root/'executions').rglob(basename))
    if include_sources:
        for folder in ('scripts', 'tests'):
            names.update(str(path.relative_to(root)) for path in (root/folder).glob('*.py'))
    names.update(extra_paths)
    selected = []
    for name in sorted(names):
        path = safe_path(root, name)
        if not allowed(name, include_sources) or not path.is_file():
            raise ValueError('Missing or non-allowlisted closeout input: '+name)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Closeout file exceeds bounded metadata size: '+name)
        selected.append(name)
    return selected


def verify_receipts(root, captured=None):
    """Every existing record must match its last independently verified bytes."""
    root = Path(root).resolve()
    def read(name):
        return captured[name] if captured is not None else safe_path(root, name).read_bytes()
    name = 'verification/verification_status.json'; raw = read(name); status = json.loads(raw)
    if status['status'] not in {'PASS', 'FAIL_INCOMPLETE'} or status['all_present_receipts_recomputed'] is not True:
        raise ValueError('Last verification does not independently validate every present receipt')
    actual = sorted(str(path.relative_to(root)) for path in (root/'records').glob('*.json'))
    rows = status['receipt_checks']
    if not actual or len(rows) != len(actual) or {row['path'] for row in rows} != set(actual):
        raise ValueError('Receipt inventory differs from last verification')
    hashes = {}
    for row in rows:
        if row['status'] != 'PASS' or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')):
            raise ValueError('Receipt lacks an independently verified PASS hash')
        if sha(read(row['path'])) != row['sha256']:
            raise ValueError('Receipt bytes differ from verifier PASS: '+row['path'])
        hashes[row['path']] = row['sha256']
    return {'path':name, 'sha256':sha(raw), 'checked_at_utc':status['checked_at_utc'],
            'overall_status':status['status'], 'receipt_sha256':dict(sorted(hashes.items()))}



def credential_content(name, data):
    if TOKEN_LITERAL.search(data):
        return True
    if PurePosixPath(name).suffix != '.json':
        return False
    protected = {'password', 'passwd', 'api_key', 'access_token', 'refresh_token', 'client_secret',
                 'private_key', 'modal_token_id', 'modal_token_secret', 'hf_token',
                 'huggingface_token', 'openai_api_key', 'aws_secret_access_key'}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized=str(key).lower().replace('-', '_')
                if normalized in protected and isinstance(item,str) and item.strip():
                    return True
                if normalized == 'authorization' and isinstance(item,str) and item.lower().startswith('bearer '):
                    return True
                if visit(item):return True
        elif isinstance(value,list):
            return any(visit(item) for item in value)
        return False
    return visit(json.loads(data))

def capture(root, extra_paths=(), include_sources=False):
    names = select_paths(root, extra_paths, include_sources)
    captured = {}
    for name in names:
        data = safe_path(root, name).read_bytes()
        if len(data) > MAX_FILE_BYTES or credential_content(name, data):
            raise ValueError('Oversized or credential-like content excluded: '+name)
        captured[name] = data
    if sum(map(len, captured.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Closeout metadata exceeds total size bound')
    verification = verify_receipts(root, captured)
    assert_current(root, captured, extra_paths, include_sources)
    return captured, verification


def assert_current(root, captured, extra_paths, include_sources):
    if select_paths(root, extra_paths, include_sources) != sorted(captured):
        raise ValueError('Closeout input inventory changed during capture')
    for name, data in captured.items():
        if sha(safe_path(root, name).read_bytes()) != sha(data):
            raise ValueError('Closeout input changed during capture: '+name)


def build(root, timestamp, output, *, extra_paths=(), include_sources=False):
    root = Path(root).resolve()
    if not re.fullmatch(r'\d{8}T\d{6}Z', timestamp):
        raise ValueError('Timestamp must have UTC form YYYYMMDDTHHMMSSZ')
    datetime.strptime(timestamp, '%Y%m%dT%H%M%SZ')
    output = Path(output)
    if not output.is_absolute():
        output = root/output
    try:
        name = str(output.relative_to(root))
    except ValueError as exc:
        raise ValueError('Output must remain inside this sprint') from exc
    output = safe_path(root, name)
    if len(PurePosixPath(name).parts) != 2 or PurePosixPath(name).parts[0] != 'closeout_packages':
        raise ValueError('Use a fresh closeout_packages/<name> output within the sprint')
    if output.exists():
        raise FileExistsError('Closeout output must be fresh; no overwrites')
    captured, verification = capture(root, extra_paths, include_sources)
    manifest = {'schema':'review_closeout_manifest_v1', 'namespace':NS, 'timestamp_utc':timestamp,
        'verification':verification, 'program_completion_claim':verification['overall_status']=='PASS',
        'scope':'Review snapshot only; raw streams, activations, models, adapters and environments are excluded and retain their original volume paths.',
        'source_snapshot_included':include_sources, 'explicit_extra_paths':sorted(set(extra_paths)),
        'files':[{'path':name, 'size_bytes':len(data), 'sha256':sha(data)} for name,data in sorted(captured.items())]}
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_bytes(json_bytes(manifest))
    with tarfile.open(output/'snapshot.tar', 'x', format=tarfile.PAX_FORMAT) as bundle:
        for name, data in sorted(captured.items()):
            info=tarfile.TarInfo(name); info.size=len(data); info.mtime=0
            info.uid=info.gid=0; info.uname=info.gname=''; info.mode=0o444
            bundle.addfile(info, BytesIO(data))
    assert_current(root, captured, extra_paths, include_sources)
    package={'schema':'review_closeout_package_v1','namespace':NS,'timestamp_utc':timestamp,
        'remote_directory':f'{NS}/closeout/{timestamp}',
        'manifest_sha256':sha((output/'manifest.json').read_bytes()),
        'bundle_sha256':sha((output/'snapshot.tar').read_bytes()),
        'bundle_size_bytes':(output/'snapshot.tar').stat().st_size,
        'overall_verification_status':verification['overall_status'], 'upload_performed':False}
    (output/'package.json').write_bytes(json_bytes(package))
    validate_package(output)
    return package


def validate_package(output):
    output=Path(output)
    package=json.loads((output/'package.json').read_bytes())
    manifest_bytes=(output/'manifest.json').read_bytes()
    if sha(manifest_bytes) != package['manifest_sha256']:
        raise ValueError('Closeout manifest changed')
    manifest=json.loads(manifest_bytes)
    if manifest['namespace'] != NS or package['namespace'] != NS:
        raise ValueError('Unexpected closeout namespace')
    timestamp=manifest['timestamp_utc']
    if not re.fullmatch(r'\d{8}T\d{6}Z',timestamp) or package['timestamp_utc'] != timestamp or package['remote_directory'] != f'{NS}/closeout/{timestamp}':
        raise ValueError('Closeout destination identity differs')
    if manifest.get('schema') != 'review_closeout_manifest_v1' or package.get('schema') != 'review_closeout_package_v1':
        raise ValueError('Unsupported closeout schema')
    status=manifest['verification']['overall_status']
    if status not in {'PASS','FAIL_INCOMPLETE'} or package['overall_verification_status'] != status or manifest['program_completion_claim'] is not (status=='PASS'):
        raise ValueError('Closeout completion claim differs from verification')
    bundle=(output/'snapshot.tar').read_bytes()
    if len(bundle) != package['bundle_size_bytes'] or sha(bundle) != package['bundle_sha256']:
        raise ValueError('Closeout bundle changed')
    expected={row['path']:row for row in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate manifest member')
    with tarfile.open(fileobj=BytesIO(bundle), mode='r:') as archive:
        members=archive.getmembers()
        if sorted(item.name for item in members) != sorted(expected):
            raise ValueError('Bundle inventory differs from manifest')
        for item in members:
            safe_path(output, item.name)
            if not item.isfile() or not allowed(item.name, manifest['source_snapshot_included']):
                raise ValueError('Unsafe bundle member')
            data=archive.extractfile(item).read()
            if len(data) != expected[item.name]['size_bytes'] or sha(data) != expected[item.name]['sha256']:
                raise ValueError('Bundle member hash differs')
    return package, manifest


def upload(root, output, *, volume=None, not_found_errors=None):
    """Explicit caller authorization required; never invoked by default CLI."""
    root=Path(root).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(root/'closeout_packages'):
        raise ValueError('Package must remain in this sprint closeout_packages directory')
    package,manifest=validate_package(output)
    package_bytes={name:(output/name).read_bytes() for name in ('manifest.json','snapshot.tar','package.json')}
    if verify_receipts(root) != manifest['verification']:
        raise ValueError('Last verifier result changed since closeout packaging')
    # Require all local inputs to remain exactly those reviewed at packaging.
    current, verification=capture(root,manifest['explicit_extra_paths'],manifest['source_snapshot_included'])
    if [{'path':name,'size_bytes':len(data),'sha256':sha(data)} for name,data in sorted(current.items())] != manifest['files']:
        raise ValueError('Local evidence changed since closeout packaging')
    if volume is None:
        import modal
        volume=modal.Volume.from_name(VOLUME)
        not_found_errors=(modal.exception.NotFoundError,FileNotFoundError)
    if not_found_errors is None:
        not_found_errors=(FileNotFoundError,)
    remote=package['remote_directory']
    try:
        volume.listdir(remote)
    except not_found_errors:
        pass
    else:
        raise FileExistsError('Remote closeout namespace already exists; no overwrites')
    with volume.batch_upload(force=False) as batch:
        for name in ('manifest.json','snapshot.tar','package.json'):
            batch.put_file(output/name,remote+'/'+name)
    verified={}
    for name in ('manifest.json','snapshot.tar','package.json'):
        digest=hashlib.sha256(); size=0
        for block in volume.read_file(remote+'/'+name):
            digest.update(block);size+=len(block)
        data=package_bytes[name]
        if (output/name).read_bytes() != data:
            raise ValueError('Local package changed during upload: '+name)
        if digest.hexdigest() != sha(data) or size != len(data):
            raise ValueError('Remote readback differs: '+name)
        verified[name]={'sha256':digest.hexdigest(),'size_bytes':size}
    validate_package(output)
    assert_current(root,current,manifest['explicit_extra_paths'],manifest['source_snapshot_included'])
    receipt={'schema':'review_closeout_readback_v1','remote_directory':'/artifacts/'+remote,
        'volume':VOLUME,'verified_files':verified,'overall_verification_status':verification['overall_status'],
        'program_completion_claim':manifest['program_completion_claim'],'upload_performed':True,
        'all_uploaded_bytes_readback_verified':True}
    completion=json_bytes(receipt)
    marker=output/'CLOSEOUT_VERIFIED.json'
    with marker.open('xb') as handle:handle.write(completion)
    with volume.batch_upload(force=False) as batch:
        batch.put_file(marker,remote+'/CLOSEOUT_VERIFIED.json')
    if b''.join(volume.read_file(remote+'/CLOSEOUT_VERIFIED.json')) != completion:
        raise ValueError('Remote completion marker readback differs')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timestamp',required=True,help='UTC YYYYMMDDTHHMMSSZ')
    parser.add_argument('--output',required=True,help='Fresh closeout_packages/<name> inside this sprint')
    parser.add_argument('--include',action='append',default=[],help='Explicit allowlisted metadata path')
    parser.add_argument('--include-sources',action='store_true',help='Include current scripts/*.py and tests/*.py')
    parser.add_argument('--upload',action='store_true',help='Explicitly upload and read back the fresh snapshot')
    args=parser.parse_args()
    result=build(ROOT,args.timestamp,args.output,extra_paths=args.include,include_sources=args.include_sources)
    if args.upload:result=upload(ROOT,ROOT/args.output if not Path(args.output).is_absolute() else args.output)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()

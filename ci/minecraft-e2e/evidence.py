#!/usr/bin/env python3
import argparse
import hashlib
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_event(root: Path, event, **fields):
    root.mkdir(parents=True, exist_ok=True)
    record = {
        'schema_version': '1.0',
        'timestamp': now(),
        'run_id': os.getenv('GITHUB_RUN_ID'),
        'run_attempt': int(os.getenv('GITHUB_RUN_ATTEMPT', '1')),
        'commit_sha': os.getenv('GITHUB_SHA'),
        'producer': fields.pop('producer', 'github-actions'),
        'event': event,
        **fields,
    }
    with (root / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')


def init(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'environment.json', {
        'schema_version': '1.0',
        'created_at': now(),
        'repository': os.getenv('GITHUB_REPOSITORY'),
        'ref': os.getenv('GITHUB_REF'),
        'commit_sha': os.getenv('GITHUB_SHA'),
        'run_id': os.getenv('GITHUB_RUN_ID'),
        'run_attempt': int(os.getenv('GITHUB_RUN_ATTEMPT', '1')),
        'runner_os': os.getenv('RUNNER_OS'),
        'runner_arch': os.getenv('RUNNER_ARCH'),
        'workflow': os.getenv('GITHUB_WORKFLOW'),
    })
    atomic_json(root / 'failure.json', {
        'schema_version': '1.0', 'status': 'none', 'failure_id': None,
        'phase': None, 'retryable': False, 'message': None, 'recorded_at': None,
    })
    append_event(root, 'run_started', phase='preflight', severity='info')


def fail(root: Path, failure_id: str, phase: str, message: str, retryable: bool):
    path = root / 'failure.json'
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            existing = {}
    if existing.get('status') != 'failed':
        data = {
            'schema_version': '1.0', 'status': 'failed', 'failure_id': failure_id,
            'phase': phase, 'retryable': retryable, 'message': message, 'recorded_at': now(),
        }
        atomic_json(path, data)
        append_event(root, 'root_failure_recorded', phase=phase, severity='error',
                     failure_id=failure_id, retryable=retryable, message=message)
    else:
        append_event(root, 'secondary_failure', phase=phase, severity='error',
                     failure_id=failure_id, retryable=retryable, message=message)


def load_oracles(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if 'oracle' in data and 'status' in data:
                return {str(data['oracle']): str(data['status'])}
            return data
    except Exception:
        pass
    # Client GameTest writers may append one compact oracle record per line.
    # Normalize that stream into the stable object consumed by the finalizer.
    merged = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict) and row.get('oracle') and row.get('status'):
            merged[str(row['oracle'])] = str(row['status'])
    return merged


def manifest(root: Path):
    files = []
    manifest_path = root / 'manifest.json'
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == manifest_path or path.name.endswith('.tmp'):
            continue
        raw = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        mime, _ = mimetypes.guess_type(rel)
        files.append({
            'path': rel, 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'mime': mime or 'application/octet-stream',
        })
    atomic_json(manifest_path, {'schema_version': '1.0', 'generated_at': now(), 'files': files})


def terminal_already_emitted(root: Path):
    path = root / 'events.jsonl'
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                if json.loads(line).get('event') == 'run_terminal':
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def finalize(root: Path, assert_pass: bool):
    failure_path = root / 'failure.json'
    failure = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {
        'status': 'failed', 'failure_id': 'PROTOCOL_FAILURE_FILE_MISSING', 'phase': 'finalize'
    }
    oracle_path = root / 'oracle-results.json'
    oracle = load_oracles(oracle_path)
    # Persist the normalized object so downstream agents never need to know
    # whether the in-process producer used an object or append-only records.
    atomic_json(oracle_path, oracle)
    required = ['l1_process', 'l2_runtime_identity', 'l3_game_state', 'l4_gpu_render']
    oracles_pass = all(oracle.get(k) == 'pass' for k in required)
    passed = failure.get('status') != 'failed' and oracles_pass
    if not passed and failure.get('status') != 'failed':
        fail(root, 'PROTOCOL_REQUIRED_ORACLE_MISSING', 'finalize',
             'Required L1-L4 oracle set is incomplete', False)
        failure = json.loads(failure_path.read_text(encoding='utf-8'))
    summary = {
        'schema_version': '1.0', 'completed_at': now(),
        'terminal_state': 'PASSED' if passed else 'FAILED',
        'commit_sha': os.getenv('GITHUB_SHA'), 'run_id': os.getenv('GITHUB_RUN_ID'),
        'run_attempt': int(os.getenv('GITHUB_RUN_ATTEMPT', '1')),
        'required_oracles': required, 'oracles': oracle, 'failure': failure,
    }
    atomic_json(root / 'run-summary.json', summary)
    if not terminal_already_emitted(root):
        append_event(root, 'run_terminal', phase='finalize', severity='info' if passed else 'error',
                     terminal_state=summary['terminal_state'], failure_id=failure.get('failure_id'))
    manifest(root)
    if assert_pass and not passed:
        raise SystemExit(79)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init')
    e = sub.add_parser('event')
    e.add_argument('--event', required=True); e.add_argument('--phase', required=True)
    e.add_argument('--severity', default='info'); e.add_argument('--message', default='')
    f = sub.add_parser('fail')
    f.add_argument('--failure-id', required=True); f.add_argument('--phase', required=True)
    f.add_argument('--message', required=True); f.add_argument('--retryable', action='store_true')
    sub.add_parser('manifest')
    z = sub.add_parser('finalize'); z.add_argument('--assert-pass', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    if a.cmd == 'init': init(root)
    elif a.cmd == 'event': append_event(root, a.event, phase=a.phase, severity=a.severity, message=a.message)
    elif a.cmd == 'fail': fail(root, a.failure_id, a.phase, a.message, a.retryable)
    elif a.cmd == 'manifest': manifest(root)
    elif a.cmd == 'finalize': finalize(root, a.assert_pass)


if __name__ == '__main__':
    main()

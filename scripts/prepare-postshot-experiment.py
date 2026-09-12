"""Explicit, audited trajectory-QA exception; never modifies a persisted Run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gsdb.manifests import load_model
from gsdb.models import RunManifest, StageStatus
from gsdb.paths import ensure_within
from gsdb import postshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--attempt', choices=['primary', 'fallback'], required=True)
    parser.add_argument('--label', required=True, help='Unique experiment directory name')
    parser.add_argument('--allow-failed-trajectory', action='store_true', required=True)
    parser.add_argument('--reason', required=True, help='Record the user authorization and purpose')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.label):
        parser.error('--label must contain only ASCII letters, digits, underscores or hyphens')
    if not args.reason.strip():
        parser.error('--reason must not be empty')
    work = args.run_dir.resolve(strict=True)
    scene = work.parent
    manifest_path = work / 'manifest.yaml'
    run = load_model(manifest_path, RunManifest).model_copy(deep=True)
    if run.id != work.name:
        raise ValueError('Run ID and directory name disagree')
    if run.config.schema_version != 4:
        raise ValueError('This exception workflow is validated only for schema 4')
    source = work / f'reconstruction-{args.attempt}'
    protected = [manifest_path, source / 'trajectory-qa.json',
                 source / 'mask-filter.json', source / 'mask-final.json']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    root = ensure_within(work / 'postshot-experiments' / args.label, work)
    root.mkdir(parents=True, exist_ok=True)
    audit = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-audit.json')
    record = dict(test_only=True, trajectory_qa_skipped=True, reason=args.reason,
                  source_hashes=before, attempt=args.attempt, status='preparing')
    audit.write_text(json.dumps(record, indent=2), encoding='utf-8')
    (root / 'TEST-ONLY.txt').write_text(
        'Trajectory QA bypassed by explicit user authorization. Not quality accepted.\n'
        'dataset.json validation refers only to dataset integrity.\n' + args.reason + '\n',
        encoding='utf-8',
    )
    # Adapt a private object to the existing preparation API; never call save_run.
    run.selected_dataset = source.relative_to(scene).as_posix()
    run.fallback_attempted = args.attempt == 'fallback'
    run.stages['reconstruct'].status = StageStatus.SUCCEEDED
    try:
        with patch.object(postshot, 'validate_trajectory_qa', return_value=None):
            result = postshot.prepare_postshot_dataset(
                scene, run, output=root / 'dataset', resume=args.resume,
            )
        record.update(status='prepared', dataset=result['output_path'], counts=result['counts'])
        (root / 'dataset' / 'TEST-ONLY.txt').write_text(
            (root / 'TEST-ONLY.txt').read_text(encoding='utf-8'), encoding='utf-8')
        print(json.dumps(record, indent=2), flush=True)
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record['source_records_unchanged'] = all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == value
            for p, value in before.items())
        audit.write_text(json.dumps(record, indent=2), encoding='utf-8')
        if not record['source_records_unchanged']:
            raise RuntimeError('Source records changed concurrently; inspect before continuing')


if __name__ == '__main__':
    main()

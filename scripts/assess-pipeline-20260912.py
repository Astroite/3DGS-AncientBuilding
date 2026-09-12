"""Read-only assessment of run evidence; writes only the requested summary file."""
from pathlib import Path
import collections
import json
import re
import sys
import numpy as np
import yaml

root = Path(__file__).resolve().parents[2]
result = []
for manifest in sorted((root / 'Data').glob('*/*/*/manifest.yaml')):
    run = yaml.safe_load(manifest.read_text(encoding='utf-8'))
    work = manifest.parent
    entry = {'run': run['id'], 'scene': run['scene_id'], 'status': run['status'],
             'reconstruction_config': run['config']['reconstruction'],
             'masking_config': run['config']['masking'],
             'reconstruction_metrics': run.get('metrics', {}).get('reconstruction'),
             'stages': run.get('stages'), 'attempts': {}}
    for label in ('primary', 'fallback'):
        dataset = work / f'reconstruction-{label}'
        filter_path = dataset / 'mask-filter.json'
        if not filter_path.exists():
            continue
        filtered = json.loads(filter_path.read_text(encoding='utf-8'))
        rows = filtered['accepted'] + filtered['rejected']
        records = [json.loads(line) for line in (work / f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        timestamps = {i: r['timestamp_seconds'] for i, r in enumerate(records, 1)}
        transforms_path = dataset / 'transforms.json'
        transforms = json.loads(transforms_path.read_text(encoding='utf-8')) if transforms_path.exists() else {'frames': []}
        registered = {str(r['file_path']).replace('\\', '/').removeprefix('images/') for r in transforms['frames']}
        attempt = {'original': filtered['original_image_count'], 'current_threshold': filtered['threshold'],
                   'current_accepted': len(filtered['accepted']), 'registered': len(registered), 'thresholds': []}
        centers_by_frame = collections.defaultdict(list)
        for item in transforms['frames']:
            f = int(re.search(r'frame_(\d+)', item['file_path'])[1])
            centers_by_frame[f].append(np.array(item['transform_matrix'])[:3, 3])
        spread = {f: float(np.max(np.linalg.norm(np.array(c)-np.median(c, axis=0), axis=1))) for f,c in centers_by_frame.items()}
        attempt['same_panorama_center_spread'] = {
            'median': float(np.median(list(spread.values()))) if spread else None,
            'p95': float(np.percentile(list(spread.values()), 95)) if spread else None,
            'worst_frames': [{'frame': f, 'max_distance_from_median_center': spread[f], 'views': len(centers_by_frame[f])} for f in sorted(spread, key=spread.get, reverse=True)[:15]],
        }
        for threshold in (0.05, 0.02, 0.01, 0.005):
            kept = [r for r in rows if r['masked_fraction'] <= threshold]
            frame_counts = collections.Counter(int(re.search(r'frame_(\d+)', r['image'])[1]) for r in kept)
            bucket_counts = collections.Counter(records[f-1]['time_bucket'] for f in frame_counts)
            kept_times = sorted(timestamps[f] for f in frame_counts)
            attempt['thresholds'].append({'threshold': threshold, 'kept': len(kept),
                'temporal_samples': len(frame_counts), 'expected_samples': len(records),
                'buckets': len(bucket_counts), 'expected_buckets': len({r['time_bucket'] for r in records}),
                'min_views_per_remaining_sample': min(frame_counts.values(), default=0),
                'median_views_per_remaining_sample': float(np.median(list(frame_counts.values()))) if frame_counts else 0,
                'samples_with_fewer_than_3_views': sum(frame_counts.get(f, 0) < 3 for f in timestamps),
                'max_gap_seconds': max(np.diff(kept_times), default=0),
                'already_registered_kept': sum(r['image'] in registered for r in kept),
                'views': dict(sorted(collections.Counter(r['image'].split('/')[0] for r in kept).items()))})
        qa_path = dataset / 'trajectory-qa.json'
        if qa_path.exists():
            qa = json.loads(qa_path.read_text(encoding='utf-8'))
            samples = qa['samples']
            speeds = [float(np.linalg.norm(np.array(b['center'])-a['center']))/(b['timestamp_seconds']-a['timestamp_seconds']) for a,b in zip(samples, samples[1:])]
            median_speed = float(np.median([v for v in speeds if v > 1e-12]))
            blocks = []
            for block in qa['blocking']:
                b = dict(block)
                if 'frames' in b:
                    b['count'] = len(b['frames'])
                if b['type'] == 'teleport':
                    dt = timestamps[b['to_frame']] - timestamps[b['from_frame']]
                    b.update(dt_seconds=dt, speed=b['distance']/dt, speed_ratio=(b['distance']/dt)/median_speed)
                blocks.append(b)
            attempt['trajectory'] = {'sample_count': qa['sample_count'], 'expected_frame_count': qa['expected_frame_count'],
                'blocking': blocks, 'warning_counts': dict(collections.Counter(b['type'] for b in qa['warnings'])),
                'median_speed': median_speed, 'max_speed_ratio': max(speeds, default=0)/median_speed,
                'max_registered_gap_seconds': max((b['timestamp_seconds']-a['timestamp_seconds'] for a,b in zip(samples,samples[1:])), default=0)}
        entry['attempts'][label] = attempt
    result.append(entry)
output = Path(sys.argv[1])
output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
for entry in result:
    print(json.dumps({'run':entry['run'], 'center_spread': {k:v['same_panorama_center_spread'] for k,v in entry['attempts'].items()}, 'low_view_samples_005': {k:v['thresholds'][-1]['samples_with_fewer_than_3_views'] for k,v in entry['attempts'].items()}}, ensure_ascii=False, default=str))

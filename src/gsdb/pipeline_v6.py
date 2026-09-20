"""Schema 6 run-owned inputs and a single source-indexed local repair.

Identity rules: prepared bytes, repair plan, and merged inventories are pinned by
hashes; resume never invents a second repair round or resurrects rejected pixels.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from .manifests import canonical_hash
from .media import analyze_frames, create_temporal_subset, summarize_frame_metrics, sha256_file
from .runs import begin_stage, complete_stage, fail_stage, save_run
from .training_data import json_write
from .storage import link_or_copy


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_records(path, records):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    temporary.replace(path)


def preprocess_v6(scene, run, resume=False):
    from .sources import _dataset_from_manifest
    from .retention import safe_path, auto_cleanup
    work = scene/run.id
    if not begin_stage(run, 'preprocess', resume=resume):
        auto_cleanup(scene, run)
        return run
    save_run(scene, run)
    try:
        target = safe_path(work, run.config.prepared_relative_path)
        adoption = work/'input-adoption.json'
        if not target.exists() and adoption.exists():
            receipt = json.loads(adoption.read_text(encoding='utf-8'))
            if receipt['target'] != run.config.prepared_relative_path or receipt['input_sha256'] != run.config.input_dataset_sha256:
                raise RuntimeError('Input adoption identity changed')
            source = safe_path(scene, receipt['source'])
            parts = Path(receipt['source']).parts
            if len(parts) != 3 or parts[0] != '.preparing' or len(parts[1]) != 32:
                raise RuntimeError('Invalid private preparation path')
            candidate = _dataset_from_manifest(source)
            if candidate.dataset_sha256 != run.config.input_dataset_sha256:
                raise RuntimeError('Input adoption bytes changed')
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
        prepared = _dataset_from_manifest(safe_path(work, run.config.prepared_relative_path))
        if prepared.dataset_sha256 != run.config.input_dataset_sha256:
            raise RuntimeError('Prepared input changed')
        records = analyze_frames(list(prepared.frame_paths),
            run.config.input.selection.end_seconds-run.config.input.selection.start_seconds,
            work/'frame-metrics.jsonl', timestamps_seconds=prepared.timestamps_seconds)
        records = [{**r, 'source_frame_index': index, 'source_sha256': digest}
                   for r, index, digest in zip(records, prepared.frame_indices, prepared.frame_sha256s)]
        write_records(work/'frame-metrics.jsonl', records)
        _, selected = create_temporal_subset(prepared.path/'frames', records, work/'equirect-primary',
            run.config.input.selection.start_seconds, run.config.preprocess.selected_per_second,
            rank_limit=run.config.reconstruction.primary.temporal_rank_limit)
        for record in selected:
            if sha256_file(work/'equirect-primary'/record['file']) != record['source_sha256']:
                raise RuntimeError('Selected panorama changed; refusing unverified resume input')
        write_records(work/'selected-primary-metrics.jsonl', selected)
        run.metrics['preprocess'] = dict(candidate_frame_count=len(records), selected_frame_count=len(selected),
            candidate_fps=run.config.preprocess.candidate_fps, selected_per_second=run.config.preprocess.selected_per_second,
            candidate=summarize_frame_metrics(records), selected=summarize_frame_metrics(selected))
        run.metrics['input'] = prepared.source_probe
        run.metrics['selection'] = run.config.input.selection.model_dump(mode='json')
        complete_stage(run, 'preprocess', message=f'Prepared {len(records)} Run-owned candidates; selected {len(selected)}')
    except Exception as error:
        fail_stage(run, 'preprocess', str(error))
        save_run(scene, run)
        raise
    save_run(scene, run)
    auto_cleanup(scene, run)
    return run


def repair_plan(run, selected, candidates, report):
    from .sources import anchored_frame_indices
    selection = run.config.input.selection
    padding = run.config.reconstruction.repair_padding_seconds
    times = {i:float(r['timestamp_seconds']) for i,r in enumerate(selected, 1)}
    windows = []
    for f in set(report.get('weak_input_frames', [])) | set(report.get('missing_frames', [])):
        windows.append((times[f]-padding, times[f]+padding))
    for issue in report.get('issues', []):
        a = times[issue.get('from_frame', issue.get('frame'))]
        b = times[issue.get('to_frame', issue.get('frame'))]
        windows.append((a-padding, b+padding))
    # Coverage gaps can be absent from the pose-issue list by design.
    ordered = sorted(times.values())
    for a,b in zip(ordered, ordered[1:]):
        if b-a > run.config.segment_qa.max_gap_seconds:
            windows.append((a-padding, b+padding))
    merged = []
    for a,b in sorted((max(selection.start_seconds,a), min(selection.end_seconds,b)) for a,b in windows):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a,b])
    probe = run.config.input.source_probe
    indices, timestamps = anchored_frame_indices(probe.frame_count, probe.fps,
        selection.start_seconds, selection.end_seconds, run.config.reconstruction.repair_candidate_fps)
    seen = {int(r['source_frame_index']) for r in candidates}
    extras = [dict(source_frame_index=i, timestamp_seconds=t) for i,t in zip(indices,timestamps)
              if i not in seen and any(a <= t <= b for a,b in merged)]
    return dict(schema_version=2, attempt=1, config_hash=run.config_hash,
        primary_records_sha256=canonical_hash(selected), candidate_records_sha256=canonical_hash(candidates),
        source_sha256=run.config.input.source_sha256, windows=merged,
        candidate_fps=run.config.reconstruction.repair_candidate_fps, frames=extras,
        helper_version=run.config.input.helper_version, sdk_version=run.config.input.sdk_version)


def _export_repair(scene, run, plan):
    from .pipeline import load_capture
    from .sources import source_adapter, probe_capture_source, validate_source_fingerprints, _validate_frame
    work = scene/run.id
    capture = load_capture(scene, run.config.capture_id)
    if [r.sha256 for r in capture.source.files] != run.config.input.source_sha256 or capture.normalization != run.config.input.normalization:
        raise RuntimeError('Capture identity changed before repair')
    folder = work/'inputs'/'repair'
    (folder/'frames').mkdir(parents=True, exist_ok=True)
    probe = probe_capture_source(capture, work/'logs'/'repair-protocol')
    if any(probe.get(k) != getattr(run.config.input, k) for k in ('helper_version','sdk_version')):
        raise RuntimeError('Repair SDK/helper changed; create a new Run')
    for key in ('fps','frame_count','width','height'):
        if probe[key] != getattr(run.config.input.source_probe,key):
            raise RuntimeError(f'Repair source probe changed: {key}')
    indices = [r['source_frame_index'] for r in plan['frames']]
    outputs = [folder/'frames'/f'frame_{i:06d}.jpg' for i in range(1,len(indices)+1)]
    width = int(capture.normalization.width or probe['width'])
    height = int(capture.normalization.height or probe['height'])
    manifest = folder/'dataset.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['plan_sha256'] != canonical_hash(plan):
            raise RuntimeError('Repair extraction identity changed')
        for path, record in zip(outputs,saved['frames']):
            if _validate_frame(path,width,height) != record['sha256']:
                raise RuntimeError('Repair extraction changed')
        return outputs
    source_adapter(capture).export(capture, validate_source_fingerprints(capture), probe, indices,
        outputs, folder, work/'logs'/'repair-protocol', width, height, True)
    validate_source_fingerprints(capture)
    records = [{**r,'file':p.name,'sha256':_validate_frame(p,width,height)} for r,p in zip(plan['frames'], outputs)]
    json_write(manifest, dict(schema_version=1, plan_sha256=canonical_hash(plan), frames=records,
        helper_version=probe.get('helper_version'), sdk_version=probe.get('sdk_version')))
    return outputs


def merge_repair_inputs(work, run, selected, extra_records, plan):
    """Remap both accepted and rejected inventories; never resurrect rejected pixels."""
    from .masking import validate_mask_filter
    from .mask_finalize import validate_mask_finalization, finalize_mask_dataset
    from .mask_review import MaskReviewDataset
    from .segments import frame_id
    dataset = work/'reconstruction-repair'
    all_records = [(r,'primary',i) for i,r in enumerate(selected,1)] + [(r,'new',i) for i,r in enumerate(extra_records,1)]
    all_records.sort(key=lambda item:item[0]['timestamp_seconds'])
    mapping = {(kind, old):i for i,(_,kind,old) in enumerate(all_records,1)}
    records = [{**r,'file':f'frame_{i:06d}.jpg','origin':kind,'origin_frame':old}
               for i,(r,kind,old) in enumerate(all_records,1)]
    identity = dict(plan_sha256=canonical_hash(plan), records_sha256=canonical_hash(records), sources={})
    inventories = {}
    for label,source in [('primary',work/'reconstruction-primary'),('new',work/'repair-new')]:
        inventories[label] = validate_mask_filter(source, verify_hashes=True)
        identity['sources'][label] = sha256_file(source/'mask-filter.json')
    receipt = dataset/'repair-input.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Repair input identity changed')
        validate_mask_filter(dataset, verify_hashes=True)
        return records
    dataset.mkdir(exist_ok=True)
    (dataset/'images').mkdir(exist_ok=True)
    (dataset/'masks').mkdir(exist_ok=True)
    merged = dict(schema_version=1, status='complete', threshold=run.config.masking.mask_discard_threshold,
                  comparison='masked_fraction > threshold', original_image_count=0, accepted=[], rejected=[])
    excluded, storage = set(), []
    for label, source in [('primary',work/'reconstruction-primary'),('new',work/'repair-new')]:
        filtered = inventories[label]
        original_excluded = set()
        if (source/'mask-final.json').exists():
            final = validate_mask_finalization(source, {r['image'] for r in filtered['accepted']})
            original_excluded = set(final['excluded_images'])
        def remap(name):
            index = mapping[label, frame_id(name)]
            return re.sub(r'frame_\d+',f'frame_{index:06d}',name)
        for kind in ('accepted','rejected'):
            for row in filtered[kind]:
                changed = {**row,'image':remap(row['image']),'mask':remap(row['mask'])}
                merged[kind].append(changed)
                if kind == 'accepted':
                    for folder,key in [('images','image'),('masks','mask')]:
                        storage.append(link_or_copy(source/folder/row[key],dataset/folder/changed[key]))
                    if row['image'] in original_excluded:
                        excluded.add(changed['image'])
        merged['original_image_count'] += filtered['original_image_count']
    for kind in ('accepted','rejected'):
        merged[kind].sort(key=lambda r:r['image'])
    json_write(dataset/'mask-filter.json',merged)
    validate_mask_filter(dataset, verify_hashes=True)
    if excluded:
        review = MaskReviewDataset(dataset)
        for item in review.items:
            if item.source_image in excluded:
                review.update_review(item.image_id,'exclude','Inherited from finalized source attempt')
    if merged['accepted'] and not run.config.masking.mask_review_required:
        finalize_mask_dataset(dataset,{r['image'] for r in merged['accepted']},run.config.masking.mask_discard_threshold)
    json_write(dataset/'storage.json',dict(files=storage,duplicate_bytes=sum(r['duplicate_bytes'] for r in storage)))
    write_records(work/'selected-repair-metrics.jsonl', records)
    json_write(receipt,identity)
    return records


def prepare_repair(scene, run, selected, report):
    from .pipeline import _prepare_masked_dataset
    from .masking import validate_mask_filter
    work = scene/run.id
    candidates = read_records(work/'frame-metrics.jsonl')
    plan = repair_plan(run,selected,candidates,report)
    path = work/'repair-plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Repair plan identity changed; create a new Run')
    else:
        json_write(path,plan)
    if not plan['frames']:
        return None
    dataset = work/'reconstruction-repair'
    if (dataset/'repair-input.json').exists():
        receipt = json.loads((dataset/'repair-input.json').read_text(encoding='utf-8'))
        records = read_records(work/'selected-repair-metrics.jsonl')
        if receipt['plan_sha256'] != canonical_hash(plan) or receipt['records_sha256'] != canonical_hash(records):
            raise RuntimeError('Committed repair dataset identity changed')
        record_repair_metrics(run, validate_mask_filter(dataset,verify_hashes=True))
        return records
    outputs = _export_repair(scene,run,plan)
    extras = analyze_frames(outputs,run.config.input.selection.end_seconds-run.config.input.selection.start_seconds,
        work/'repair-frame-metrics.jsonl',timestamps_seconds=[r['timestamp_seconds'] for r in plan['frames']])
    extras = [{**r,**identity,'source_file':r['file'],'source_sha256':sha256_file(p)}
              for r,identity,p in zip(extras,plan['frames'],outputs)]
    metrics = _prepare_masked_dataset(scene,work,outputs[0].parent,work/'repair-new',
        run.config.reconstruction.primary,run,'repair-new')
    run.metrics.setdefault('masking',{})['repair-new'] = metrics
    records = merge_repair_inputs(work,run,selected,extras,plan)
    write_records(work/'selected-repair-metrics.jsonl',records)
    record_repair_metrics(run, validate_mask_filter(dataset,verify_hashes=True))
    return records


def record_repair_metrics(run, filtered):
    run.metrics.setdefault('masking',{})['repair'] = dict(projected_planar_images=filtered['original_image_count'],
        planar_images=filtered['original_image_count'],
        reconstruction_input_images=len(filtered['accepted']), automatic_rejected_images=len(filtered['rejected']),
        automatic_rejected_fraction=len(filtered['rejected'])/filtered['original_image_count'] if filtered['original_image_count'] else 0.)

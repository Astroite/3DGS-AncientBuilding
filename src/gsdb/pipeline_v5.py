"""One robust reconstruction followed by at most one targeted repair dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .manifests import canonical_hash
from .processes import CommandError
from .runs import begin_stage, complete_stage, fail_stage, save_run
from .segments import analyze_segments, repair_records, write_segments


def reconstruct_v5(scene: Path, run, resume=False):
    from .pipeline import _prepare_masked_dataset, _v4_attempt_inventory
    from .masking import validate_mask_filter
    from .reconstruction_realityscan import run_realityscan_alignment
    from .sources import _dataset_from_manifest

    if not begin_stage(run,'reconstruct',resume=resume):
        from .retention import auto_cleanup
        auto_cleanup(scene, run)
        return run
    work = scene/run.id
    save_run(scene,run)
    try:
        selected = [json.loads(line) for line in (work/'selected-primary-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        attempts = run.metrics.setdefault('reconstruction',{})
        viable = []
        for label, records in [('primary',selected),('repair',None)]:
            if label == 'repair':
                if getattr(run.config, 'schema_version', 5) == 6:
                    from .pipeline_v6 import prepare_repair
                    records = prepare_repair(scene, run, selected, report)
                    if records is None:
                        break
                else:
                    prepared = _dataset_from_manifest(scene/run.config.prepared_relative_path)
                    if prepared.dataset_sha256 != run.config.input_dataset_sha256:
                        raise RuntimeError('Prepared input changed before repair')
                    candidates = [json.loads(line) for line in (work/'frame-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
                    extra = repair_records(candidates,selected,report,run.config.reconstruction.repair_padding_seconds)
                    if not extra:
                        break
                    merged = sorted(selected+extra,key=lambda r:float(r['timestamp_seconds']))
                    decision = {'schema_version':1,'attempt':1,'source_records_sha256':canonical_hash(selected),
                                'added_source_files':[r['source_file'] for r in extra]}
                    decision_path = work/'repair-plan.json'
                    if decision_path.exists() and json.loads(decision_path.read_text(encoding='utf-8')) != decision:
                        raise RuntimeError('Repair inputs changed; create a new run')
                    decision_path.write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
                    source_dir = work/'equirect-repair'
                    source_dir.mkdir(exist_ok=True)
                    records = []
                    for index, r in enumerate(merged,1):
                        source_name = r.get('source_file',r['file'])
                        name = f'frame_{index:06d}.jpg'
                        target = source_dir/name
                        original = prepared.path/'frames'/source_name
                        if not target.exists():
                            os.link(original,target)
                        elif not os.path.samefile(original,target):
                            raise RuntimeError(f'Repair source conflict: {target}')
                        records.append({**r,'source_file':source_name,'file':name})
                    (work/'selected-repair-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
                    masking = _prepare_masked_dataset(scene,work,source_dir,work/'reconstruction-repair',
                                                     run.config.reconstruction.primary,run,'repair')
                    run.metrics.setdefault('masking',{})['repair'] = masking
                run.fallback_attempted = True
                save_run(scene,run)
                if getattr(run.config, 'schema_version', 5) == 6:
                    from .retention import auto_cleanup
                    auto_cleanup(scene, run, checkpoint=True)
            dataset = work/f'reconstruction-{label}'
            filtered = validate_mask_filter(dataset,verify_hashes=False)
            included = {r['image'] for r in filtered['accepted']}
            if included:
                _,included = _v4_attempt_inventory(dataset,run)
            if len(included) >= 2:
                try:
                    metrics = run_realityscan_alignment(dataset,run.config.reconstruction.primary,
                        work/'logs'/f'reconstruct-{label}',run.config.reconstruction,
                        included_images=included,projected_image_count=filtered['original_image_count'])
                except CommandError as error:
                    if label!='repair' or not viable:
                        raise
                    attempts[label]={'status':'failed','error':str(error),'retained_valid_primary':True}
                    break
                attempts[label] = metrics
                report = write_segments(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa)
            else:
                report = analyze_segments([],records,included,run.config.segment_qa,[])
                attempts[label] = {'registered_images':0,'reason':'fewer than two included images'}
                report['lineage']={'records_sha256':canonical_hash(records),'included_sha256':canonical_hash(sorted(included))}
                temporary=dataset/'segments.json.tmp'
                temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
                temporary.replace(dataset/'segments.json')
            run.metrics['segment_qa'] = {'attempt':label,'training_status':report['training_status'],
                'coverage_status':report['coverage_status'],'report':f'{run.id}/reconstruction-{label}/segments.json'}
            if report['training_status']=='passed':
                # Keep independently aligned models separate. Prefer covered time,
                # then usable images; do not discard valid primary segments when
                # the single repair attempt yields a worse component.
                passed = [s for s in report['segments'] if s['status']=='passed']
                score = (sum(s['end_seconds']-s['start_seconds'] for s in passed),sum(len(s['images']) for s in passed))
                viable.append((score,dataset,report,label))
            save_run(scene,run)
        if not viable:
            raise RuntimeError('No valid training segment after the bounded repair attempt')
        _,dataset,report,label = max(viable,key=lambda item:item[0])
        run.metrics['segment_qa'] = {'attempt':label,'training_status':report['training_status'],
            'coverage_status':report['coverage_status'],'report':f'{run.id}/reconstruction-{label}/segments.json'}
        run.selected_dataset = dataset.relative_to(scene).as_posix()
        run.metrics['selected_attempt'] = dataset.name.removeprefix('reconstruction-')
        complete_stage(run,'reconstruct',message=f"Valid segments available; route coverage {report['coverage_status']}")
        save_run(scene,run)
        from .retention import auto_cleanup
        auto_cleanup(scene, run)
        return run
    except Exception as error:
        fail_stage(run,'reconstruct',str(error))
        if getattr(run.config, 'schema_version', 5) == 6:
            from .mask_finalize import MaskFinalizationMissingError
            from .models import RunStatus
            if isinstance(error, MaskFinalizationMissingError):
                run.status = RunStatus.WAITING_REVIEW
        save_run(scene,run)
        raise

"""Single-profile reconstruction controls, isolated from historical Runs."""
import json

from .gpu_lock import gpu_locked
from .masking import validate_mask_filter
from .mask_finalize import validate_mask_finalization
from .reconstruction_realityscan import run_realityscan_alignment,realityscan_reconstruction_metrics
from .runs import begin_stage,complete_stage,fail_stage,save_run
from .segments import write_segments,segment_report


@gpu_locked
def reconstruct_profile_baseline(scene,run,resume=False):
    """Evaluate just one historical projection profile with the new common QA."""
    if not begin_stage(run,'reconstruct',resume=resume):
        return run
    work=scene/run.id
    dataset=work/'reconstruction-primary'
    save_run(scene,run)
    try:
        filtered=validate_mask_filter(dataset,verify_hashes=True)
        included={r['image'] for r in filtered['accepted']}
        final=validate_mask_finalization(dataset,included)
        included-=set(final.get('excluded_images',[]))
        records=[json.loads(line) for line in (work/'selected-primary-metrics.jsonl').read_text().splitlines() if line]
        report_path=dataset/'segments.json'
        if report_path.is_file():
            report=segment_report(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa)
            if report!=json.loads(report_path.read_text(encoding='utf-8')):
                raise RuntimeError('Baseline evidence changed; create a new benchmark Run')
            metrics=realityscan_reconstruction_metrics(dataset,run.config.reconstruction.primary,run.config.reconstruction,
                included_images=included,projected_image_count=filtered['original_image_count'])
        else:
            metrics=run_realityscan_alignment(dataset,run.config.reconstruction.primary,work/'logs'/'reconstruct-primary',
                run.config.reconstruction,included_images=included,projected_image_count=filtered['original_image_count'])
            report=write_segments(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa)
        run.metrics['reconstruction']={'primary':metrics}
        run.metrics['comparison_policy']='Single historical profile; targeted repair intentionally not invoked for this control'
        run.metrics['segment_qa']=dict(attempt='primary',training_status=report['training_status'],
            coverage_status=report['coverage_status'],report=f'{run.id}/reconstruction-primary/segments.json')
        if report['training_status']!='passed':
            raise RuntimeError('Historical-profile control has no passing training segment')
        run.selected_dataset=dataset.relative_to(scene).as_posix()
        run.metrics['selected_attempt']='primary'
        complete_stage(run,'reconstruct',message=f'Historical-profile control; coverage {report["coverage_status"]}')
    except Exception as error:
        fail_stage(run,'reconstruct',str(error))
        save_run(scene,run)
        raise
    save_run(scene,run)
    return run

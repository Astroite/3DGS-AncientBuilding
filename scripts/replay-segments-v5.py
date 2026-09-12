"""Evaluate legacy poses with new QA into a separate directory; preserve source Runs."""
from pathlib import Path
import argparse
import json
from collections import Counter

from gsdb.media import sha256_file
from gsdb.models import SegmentQAConfig
from gsdb.runs import load_run
from gsdb.segments import write_segments
from gsdb.training_data import json_write

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
summaries=[]
for path in sorted(args.data.glob('*/*/*/manifest.yaml')):
    before=sha256_file(path)
    run=load_run(path.parent.parent,path.parent.name)
    if run.config.schema_version != 4:
        continue
    summary={'run':run.id,'scene':run.scene_id,'source_manifest_sha256':before,'attempts':{}}
    for label in ['primary','fallback']:
        dataset=path.parent/f'reconstruction-{label}'
        if not (dataset/'transforms.json').exists():
            continue
        filtered=json.loads((dataset/'mask-filter.json').read_text(encoding='utf-8'))
        final=json.loads((dataset/'mask-final.json').read_text(encoding='utf-8'))
        included={r['image'] for r in filtered['accepted']}-set(final.get('excluded_images',[]))
        records=[json.loads(line) for line in (path.parent/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        output=args.output/run.id/label/'segments.json'
        report=write_segments(dataset,records,included,getattr(run.config.reconstruction,label),SegmentQAConfig(),output)
        stable=[s for s in report['segments'] if s['status']=='passed' and s['end_seconds']-s['start_seconds']>=30]
        summary['attempts'][label]={'coverage':report['coverage_status'],'training':report['training_status'],
            'issues':dict(Counter(i['type'] for i in report['issues'])),'passed_segments':sum(s['status']=='passed' for s in report['segments']),
            'first_stable_30_seconds':[stable[0]['start_seconds'],stable[0]['start_seconds']+30] if stable else None,
            'report':str(output.resolve())}
    if sha256_file(path)!=before:
        raise RuntimeError('Source Run changed during replay')
    summaries.append(summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
args.output.mkdir(parents=True,exist_ok=True)
json_write(args.output/'summary.json',summaries)

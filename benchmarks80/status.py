"""Read saved case counts and actual process liveness; snapshots are not jobs."""
import argparse,collections,json,os
from pathlib import Path
from .common import OUT,BACKENDS,methods,manifest


def status(out=OUT):
 out=Path(out);rows=manifest(out);ready=[r for r in rows if r['status']=='ready'];valid=set(r['key'] for r in ready);counts=collections.Counter();failures=[];seen=collections.defaultdict(set)
 for p in (out/'cases').glob('*/*/seed*/*/result.json'):
  try:r=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  if r.get('dataset') not in valid:continue
  if r['status']=='ok':counts[r['dataset'],r['backend']]+=1;seen[r['dataset'],r['backend']].add(r['seed'])
  else:failures.append(str(p))
 executions={}
 for p in (out/'execution').glob('*.json'):
  try:executions[p.stem]=json.loads(p.read_text())
  except json.JSONDecodeError:continue
 running=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:name=(p/'comm').read_text().strip();args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if name.startswith('python') and b'--dry-run' not in args and any(a in [b'benchmarks80.run',b'benchmarks80.worker'] for a in args):running.append(int(p.name))
 completed=[k for k,s in executions.items() if s['status']=='completed'];deferred=[k for k,s in executions.items() if s['status']=='deferred_resource'];failed=[k for k,s in executions.items() if s['status']=='failed']
 return dict(space='H',catalog_entries=80,ready_datasets=len(ready),candidate_cases=len(ready)*350,
  stored_successful_cases=sum(counts.values()),verified_completed_datasets=len(completed),verified_completed_cases=len(completed)*350,
  by_backend={b:sum(v for (k,a),v in counts.items() if a==b) for b in BACKENDS},
  acquisition_statuses=dict(collections.Counter(r['status'] for r in rows)),deferred_resource_datasets=deferred,failed_datasets=failed,failed_case_files=failures,
  running_pids=running,actually_running=bool(running),seed_range=list(range(10)),sampling=False)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();print(json.dumps(status(a.output),indent=2))

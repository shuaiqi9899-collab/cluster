"""Finish the active queue, retry memory-quota deferrals alone, export locally."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .common import OUT,ROOT,write
from .status import status
from .snapshot import export


def finish(out):
 out=Path(out);lock=(out/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  current=status(out)
  if not current['actually_running']:break
  write(out/'finish_status.json',dict(phase='waiting_for_active_queue',updated_unix=time.time(),observed=current))
  time.sleep(20)
 state=json.loads((out/'run_status.json').read_text())
 if not state.get('phase','').startswith('completed'):
  write(out/'finish_status.json',dict(phase='interrupted_needs_resume',run_phase=state.get('phase'),updated_unix=time.time()));return
 retry=[]
 for p in (out/'execution').glob('*.json'):
  s=json.loads(p.read_text())
  if s.get('status')=='deferred_resource' and 'RSS exceeded8' in s.get('reason',''):retry.append(p.stem)
 if retry:
  write(out/'memory_retry.json',dict(reason='8GiB was a concurrency quota, not total cloud capacity; retry each memory-limited dataset alone before deferral',datasets=retry,workers=1,rss_gib=24,seeds=list(range(10)),training_protocol_unchanged=True))
  command=[sys.executable,'-u','-m','benchmarks80.run','--output',str(out),'--workers','1','--rss-gib','24','--datasets',*retry]
  with (out/'memory_retry.log').open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 export(out)
 write(out/'finish_status.json',dict(phase='local_snapshot_exported',updated_unix=time.time(),status=status(out),github_publication='Not automatic; snapshot export is local only'))
 print('Queue postprocessing finished; no remote publication performed.',flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();finish(a.output)

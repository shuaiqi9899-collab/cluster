"""Execute all ten seeds of all three backends for one admitted dataset."""
import argparse,json,os,shutil,sys,time,traceback
from pathlib import Path
from .common import OUT,BACKENDS,methods,write,training_sources
from .prepare import prepare_dataset
from .experiment import run_case
from .context import Bank
from .verify import verify_dataset


def run_dataset(out,key):
 out=Path(out);statefile=out/'execution'/(key+'.json');started=time.time()
 def state(status,**details):write(statefile,dict(dataset=key,status=status,pid=os.getpid(),started_unix=started,updated_unix=time.time(),space='H',**details))
 try:
  protocol=json.loads((out/'protocol.json').read_text())
  if protocol['training_sources']!=training_sources():raise RuntimeError('Training source hashes differ from frozen protocol')
  state('preparing',phase_started_unix=time.time());info=prepare_dataset(out,key)
  # No training case is started until all native AC families and ten SE
  # partitions have passed. Preparation is shared by every backend.
  for backend in BACKENDS:
   for seed in range(10):
    bank=Bank(out,key,backend)
    for method in methods(backend):
     if shutil.disk_usage(out).free<5*1024**3:raise MemoryError('Disk reserve below5GiB; preserve current artifacts and defer new work')
     state('training',backend=backend,seed=seed,method=method,case_started_unix=time.time())
     r=run_case(out,key,backend,seed,method,bank)
     print(key,backend,seed,method,r['status'],f'{r["seconds"]:.2f}s',flush=True)
  state('verifying',phase_started_unix=time.time());check=verify_dataset(out,key)
  state('completed',verified_cases=check['verified_cases'],support_reward_checks=check['independent_support_reward_checks'],seconds=time.time()-started)
  return 0
 except MemoryError as exc:
  state('deferred_resource',reason=str(exc),error=traceback.format_exc());return 75
 except BaseException as exc:
  state('failed',reason=str(exc),error=traceback.format_exc());raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();sys.exit(run_dataset(a.output,a.dataset))

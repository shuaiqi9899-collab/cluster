"""Resume the full ten-seed matrix, then validate and refresh all reports."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hyperbolic.prepare import ROOT,DATASETS,write_json
from .experiment import SPACES,methods_for
from .lifecycle import ACTIVE_BACKENDS,RETIRED_BACKENDS,check_training_hashes


def census(out):
    expected={(backend,space,dataset,seed,method)
              for backend in ACTIVE_BACKENDS for space in SPACES
              for dataset in DATASETS for seed in range(10) for method in methods_for(space)}
    passed=set();failed=[];counts=collections.Counter();seeds=collections.defaultdict(set)
    for path in out.glob('*/cases/*/*/result.json'):
        try:r=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        identity=(r.get('backend',path.parts[-5]),r['space'],r['dataset'],r['seed'],r['method'])
        if identity not in expected:continue
        if r['status']=='ok':
            passed.add(identity);counts[f'{identity[0]}/{r["space"]}']+=1
            seeds[identity[0]].add(r['seed'])
        else:failed.append(str(path))
    return dict(expected=len(expected),passed=len(passed),remaining=len(expected-passed),
                failed=failed,counts=dict(counts),seeds_seen={k:sorted(v) for k,v in seeds.items()},
                active_backends=list(ACTIVE_BACKENDS),retired_backends=list(RETIRED_BACKENDS))


def check_core(out):
    for path in (out/'invocations').glob('*.json'):
        record=json.loads(path.read_text())
        errors=check_training_hashes(record)
        if errors:raise RuntimeError(f'Preserve existing results: training core differs from {path.name}: {errors}')


def complete(out,workers):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'complete.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_core(out)
    status=census(out)
    if status['failed']:raise RuntimeError(f'Existing failed cases require diagnosis: {status["failed"]}')
    phases=[('training','density_cf.experiment',['--workers',str(workers)]),
            ('reward_and_artifact_verification','density_cf.verify',[]),
            ('independent_backend_audit','density_cf.audit',[]),
            ('report_and_figures','density_cf.analyze',[])]
    started=time.time()
    def save(phase,**extra):
        state=dict(census(out),phase=phase,supervisor_pid=os.getpid(),started_unix=started,
                   updated_unix=time.time(),seeds=list(range(10)),**extra)
        temp=out/'run_status.tmp.json';write_json(temp,state);temp.replace(out/'run_status.json')
        print(json.dumps(state),flush=True)
    try:
        for phase,module,arguments in phases:
            if phase=='training' and census(out)['remaining']==0:continue
            if phase=='report_and_figures':
                # Earlier readiness records truthfully described the seed0 pilot.
                # Preserve that evidence and update only the completed scope.
                readiness=out/'readiness.json'
                if readiness.exists():
                    r=json.loads(readiness.read_text());r['initial_pilot']=r.get('executed_matrix')
                    r['executed_matrix']='HDBSCAN full ten-seed matrix:2800 cases, E/H and fixed-region/mapping controls'
                    r['unrun']=[];r['retired_backends']=list(RETIRED_BACKENDS);write_json(readiness,r)
            command=[sys.executable,'-u','-m',module,'--output',str(out)]+arguments
            logfile=out/f'complete_{phase}.log'
            with logfile.open('a') as log:
                log.write('\n'+json.dumps(dict(command=command,time_unix=time.time()))+'\n');log.flush()
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    save(phase,child_pid=process.pid,log=str(logfile))
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:pass
                if process.returncode:raise RuntimeError(f'{module} failed with exit{process.returncode}; inspect {logfile}')
            status=census(out)
            if phase=='training' and (status['remaining'] or status['failed']):
                raise RuntimeError(f'Training exited without completing the declared matrix: {status}')
        save('completed')
    except BaseException as exc:
        save('failed',error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/density_cf')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.dry_run:check_core(out);print(json.dumps(census(out)))
    else:complete(out,a.workers)

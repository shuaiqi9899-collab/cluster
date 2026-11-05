"""Monitor an active matrix, then run full replay, artifact audit and analysis.

Processes do not survive cloud snapshots; rerun experiment/finish after restore.
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .prepare import ROOT


def finish(out):
    replay_done=False
    while True:
        rows=[]
        for path in (out/'cases').glob('*/*/result.json'):
            try:
                rows.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                # A worker may currently be writing its final result; retry later.
                continue
        failed=[r for r in rows if r['status']!='ok']
        if failed:
            raise RuntimeError(f'Failed cases require diagnosis: {[(r["space"],r["dataset"],r["method"],r["seed"]) for r in failed]}')
        counts=Counter(r['space'] for r in rows)
        print(time.strftime('%H:%M:%S',time.gmtime()),len(rows),'of 2640',dict(counts),flush=True)
        if counts['E']==880 and not replay_done:
            with (out/'full_replay.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-m','hyperbolic.replay','--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            replay_done=True
            print('Complete E candidate replay finished.',flush=True)
        if len(rows)==2640: break
        time.sleep(20)
    for module,logname in [('hyperbolic.verify','artifact_checks.log'),('hyperbolic.analyze','analysis.log')]:
        with (out/logname).open('w') as log:
            subprocess.run([sys.executable,'-u','-m',module,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Full matrix, replay, artifact checks and analysis completed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/hyperbolic')
    finish(p.parse_args().output.resolve())

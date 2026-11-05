"""Wait for the declared matrix, then perform complete independent analysis."""
import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path
from hyperbolic.prepare import ROOT


def finish(out,source):
    while True:
        rows=[]
        for p in (out/'cases').glob('*/*/result.json'):
            try:rows.append(json.loads(p.read_text()))
            except json.JSONDecodeError:continue
        failures=[r for r in rows if r['status']!='ok']
        if failures:raise RuntimeError(f'Failed cases need diagnosis: {failures[:1]}')
        print(time.strftime('%H:%M:%S',time.gmtime()),len(rows),'of 2800',dict(collections.Counter(r['space'] for r in rows)),flush=True)
        if len(rows)==2800:break
        time.sleep(20)
    commands=[('ac_joint.audit_ward','ward_audit.log'),('ac_joint.sweep','sweep.log'),
              ('ac_joint.verify','artifact_checks.log'),('ac_joint.analyze','analysis.log')]
    for module,logname in commands:
        args=[sys.executable,'-u','-m',module,'--output',str(out)]
        if module!='ac_joint.analyze':args+=['--source',str(source)]
        if module in ['ac_joint.sweep','ac_joint.verify']:args+=['--workers','4']
        with (out/logname).open('w') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(module,'completed.',flush=True)
    print('Complete AC matrix, Ward audit, threshold sweeps, artifact checks and report finished.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--output',type=Path,default=ROOT/'results/ac_joint')
    a=p.parse_args();finish(a.output.resolve(),a.source.resolve())

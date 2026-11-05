"""Independent Karcher-objective checks on persisted native H-Ward merges."""
import argparse
import json
from pathlib import Path
import numpy as np

from hyperbolic.geometry import frechet,ball_distance
from hyperbolic.prepare import ROOT,write_json


def audit(out,source):
    results=[]
    for index_path in sorted((out/'banks').glob('*/*/index.json')):
        index=json.loads(index_path.read_text())
        if not index['radius']:continue
        points=np.load(source/'geometry'/index['dataset']/f'r{index["radius"]:g}'/'points.npy')
        for key,paths in index['trees'].items():
            f=np.load(out/paths['ward']);z,ids,records=f['z'],f['ids'],f['merge_audit'];n=len(ids)
            if n<2:continue
            def collect(node):
                stack=[node];leaves=[]
                while stack:
                    current=stack.pop()
                    if current<n:leaves.append(current)
                    else:stack.extend([int(z[current-n,0]),int(z[current-n,1])])
                return np.asarray(sorted(leaves),dtype=int)
            memo={}
            def objective(node):
                if node not in memo:
                    leaves=collect(node);p=points[ids[leaves]]
                    q,it,res=frechet(p,tolerance=1e-7)
                    memo[node]=float(np.sum(ball_distance(q[None],p)**2))
                return memo[node]
            checks=[]
            for step in sorted(set([0,(n-2)//2,n-2])):
                a,b=int(z[step,0]),int(z[step,1]);node=n+step
                value=objective(node)-objective(a)-objective(b)
                error=abs(value-records[step,0])
                allowance=1e-8+10*records[step,1]+1e-12*n
                if error>allowance:raise AssertionError((index['dataset'],index['space'],key,step,value,records[step,0],error))
                checks.append(dict(merge=step,independent_increment=value,saved_increment=float(records[step,0]),absolute_error=error))
            results.append(dict(dataset=index['dataset'],space=index['space'],subset=key,n=n,checks=checks))
    report=dict(status='passed',native_ward_trees=len(results),checked_merge_objectives=sum(len(r['checks']) for r in results),
                solver='Independent first-order Karcher solve from prior geometry module; no Newton Hessian or linkage recursion',results=results)
    write_json(out/'validation/ward_objectives.json',report)
    print('Native Ward objective audit passed:',len(results),'trees',report['checked_merge_objectives'],'merges',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--output',type=Path,default=ROOT/'results/ac_joint')
    a=p.parse_args();audit(a.output.resolve(),a.source.resolve())

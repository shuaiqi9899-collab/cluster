"""Complete support-objective scan over all four height-cut families.

Only merges joining two support-bearing components change support NMI.
Evaluating those plateaus covers the entire threshold domain, without using K.
This is an upper reference for each local support objective, not held-out NMI.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari

from hyperbolic.prepare import ROOT,DATASETS,write_json
from .context import GeometryBank
from .trees import LINKAGES,cut_tree
from .experiment import merge


def support_scan(z,heights,n,mask,support_labels):
    groups=[[] for _ in range(2*n-1)]
    for j,i in enumerate(mask):groups[int(i)]=[j]
    prediction=np.arange(len(mask));evaluations=0;best_score=-np.inf;best_tau=0.
    order=np.lexsort((np.arange(len(heights)),heights))
    position=0
    def evaluate():return float(nmi(support_labels,prediction)) if len(mask)>=2 else 0.
    # All zero-height merges must be included at tau=0 before its evaluation.
    while position<len(order) and heights[order[position]]==0:
        i=int(order[position]);a,b=int(z[i,0]),int(z[i,1]);groups[n+i]=groups[a]+groups[b]
        if groups[n+i]:prediction[groups[n+i]]=min(groups[n+i])
        position+=1
    best_score=evaluate();evaluations+=1
    while position<len(order):
        threshold=float(heights[order[position]]);changed=False
        while position<len(order) and heights[order[position]]==threshold:
            i=int(order[position]);a,b=int(z[i,0]),int(z[i,1]);left,right=groups[a],groups[b]
            groups[n+i]=left+right
            if left and right:
                prediction[groups[n+i]]=min(groups[n+i]);changed=True
            position+=1
        if changed:
            score=evaluate();evaluations+=1
            if score>best_score:best_score,best_tau=score,threshold
    return best_score,best_tau,evaluations


def sweep_group(task):
    source,out,dataset,space,seed=task;source,out=Path(source),Path(out)
    path=out/'sweeps'/space/f'{dataset}_seed{seed}.json'
    if path.exists():return json.loads(path.read_text())
    bank=GeometryBank(source,out,dataset,space)
    frozen=np.load(source/'inputs'/dataset/f'seed{seed}.npz');supports=frozen['sample_ids'][frozen['reward_ids']]
    y=np.load(source/'inputs'/dataset/'canonical.npz')['y'];held=np.ones(len(y),bool);held[supports]=False
    scopes={'global':[bank.index['global_key']],
            'native_regions':bank.index['regions'][f'{space}_seed{seed}']}
    if space=='H':scopes['fixed_E_regions']=bank.index['regions'][f'E_seed{seed}']
    results={};start=time.perf_counter()
    for scope,keys in scopes.items():
        frames=[];parts=[];selections=[]
        for key in keys:
            trees=bank.load(key);ids=trees[0][2];mask=np.flatnonzero(np.isin(ids,supports));target=y[ids[mask]]
            best=None;evaluations=0
            for linkage_id in np.roll(np.arange(4),seed%4):
                z,h,_=trees[int(linkage_id)]
                score,tau,count=support_scan(z,h,len(ids),mask,target);evaluations+=count
                if best is None or score>best['score']:
                    best=dict(linkage=LINKAGES[int(linkage_id)],linkage_id=int(linkage_id),score=score,
                              threshold=tau,u=tau/bank.index['scales'][LINKAGES[int(linkage_id)]])
            z,h,_=trees[best['linkage_id']];frames.append(cut_tree(z,h,len(ids),best['threshold']));parts.append(ids)
            best['objective_evaluations']=evaluations;best['support_count']=len(mask);best['ids_key']=key;selections.append(best)
        prediction=merge(parts,frames,len(y))
        results[scope]=dict(heldout_nmi=float(nmi(y[held],prediction[held])),heldout_ari=float(ari(y[held],prediction[held])),
                            support_nmi=float(nmi(y[supports],prediction[supports])),selections=selections)
    result=dict(dataset=dataset,space=space,seed=seed,results=results,seconds=time.perf_counter()-start,
                objective='Independent maximum local support NMI over all four tree cut families; not a global or held-out upper bound')
    write_json(path,result);print('Sweep',space,dataset,seed,flush=True)
    return result


def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--output',type=Path,default=ROOT/'results/ac_joint');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--spaces',nargs='+',choices=['E','H'],default=['E','H'])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    a=p.parse_args();tasks=[(str(a.source.resolve()),str(a.output.resolve()),d,s,seed) for s in a.spaces for d in a.datasets for seed in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[f.result() for f in as_completed([pool.submit(sweep_group,t) for t in tasks])]
    write_json(a.output.resolve()/'sweeps/summary.json',dict(groups=len(results),results=results))


if __name__=='__main__':main()

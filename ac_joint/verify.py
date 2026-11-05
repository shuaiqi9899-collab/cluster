"""Independent cut/reward/selection checks over all persisted candidate requests."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari

from hyperbolic.prepare import ROOT,DATASETS,write_json
from .experiment import merge,methods_for,SPACES
from .context import GeometryBank
from .trees import LINKAGES


def canonical(labels):
    _,first,inverse=np.unique(labels,return_index=True,return_inverse=True)
    ranks=np.empty(len(first),dtype=int);ranks[np.argsort(first)]=np.arange(len(first))
    return ranks[inverse]


def verify_case(task):
    path,source,out=task;path=Path(path);source,out=Path(source),Path(out)
    result=json.loads(path.read_text());assert result['status']=='ok',str(path)
    bank=GeometryBank(source,out,result['dataset'],result['space'])
    folder=path.parent;dataset=result['dataset'];seed=result['seed']
    frozen=np.load(source/'inputs'/dataset/f'seed{seed}.npz')
    supports=frozen['sample_ids'][frozen['reward_ids']]
    y=np.load(source/'inputs'/dataset/'canonical.npz')['y'];parts=[];best=[]
    total_queries=total_cuts=reward_checks=cut_checks=sweep_checks=0
    sweep_space='H' if result['space']=='H-fixed-E' else result['space']
    sweep_path=out/'sweeps'/sweep_space/f'{dataset}_seed{seed}.json'
    ceiling={}
    if sweep_path.exists():
        sweep=json.loads(sweep_path.read_text())
        scope='fixed_E_regions' if result['space']=='H-fixed-E' else ('native_regions' if result['method'].startswith('MA-') else 'global')
        ceiling={v['ids_key']:v['score'] for v in sweep['results'][scope]['selections']}
    for region,info in enumerate(result['regions']):
        file=np.load(folder/f'candidates_region{region}.npz');ids=file['ids'];mask=file['support_local_ids']
        np.testing.assert_array_equal(mask,np.flatnonzero(np.isin(ids,supports)))
        events=json.loads((folder/f'events_region{region}.json').read_text())
        queries=json.loads((folder/f'queries_region{region}.json').read_text())
        labels=file['labels'];scores=file['scores'];assert len(events)==len(labels)==len(scores)
        if ceiling:
            assert float(np.max(scores))<=ceiling[info['ids_key']]+1e-12
            sweep_checks+=1
        assert len(events)==info['training_requests']
        assert len(queries)==sum(q['mode'] in ['train','detect'] for q in queries)
        assert len(events)==sum(q['mode']=='train' for q in queries)
        assert len(queries)==info['costs'].get('train_requests',0)+info['costs'].get('detect_requests',0)
        pairs={(q['linkage'],q['u']) for q in queries}
        assert len(pairs)==info['unique_parameter_pairs']==info['costs']['actual_cuts']
        assert len(events)>=4 and set(e['linkage'] for e in events[:4])==set(LINKAGES)
        if result['method'] not in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:assert len(events)==30
        checked_rewards={};checked_cuts=set();trees=bank.load(info['ids_key'])
        for event,prediction,score in zip(events,labels,scores):
            assert np.all(prediction>=0)
            linkage_id=LINKAGES.index(event['linkage']);u=event['u']
            assert 0<=u<=1
            assert event['threshold']==u*bank.index['scales'][event['linkage']]
            key=(linkage_id,prediction.tobytes())
            if key not in checked_cuts:
                z,h,treeids=trees[linkage_id];np.testing.assert_array_equal(ids,treeids)
                if len(ids)==1:reference=np.zeros(1,dtype=int)
                else:
                    copy=z.copy();copy[:,2]=h
                    reference=canonical(fcluster(copy,event['threshold'],criterion='distance'))
                np.testing.assert_array_equal(prediction,reference)
                checked_cuts.add(key);cut_checks+=1
            reward_key=prediction[mask].tobytes()
            if reward_key not in checked_rewards:
                checked_rewards[reward_key]=float(nmi(y[ids[mask]],prediction[mask])) if len(mask)>=2 else 0.
                reward_checks+=1
            assert score==event['score']==checked_rewards[reward_key]
        selected=int(np.argmax(scores[:30]));parts.append(ids);best.append(labels[selected])
        total_queries+=len(queries);total_cuts+=len(pairs)
        if 'FrozenActor' in result['method']:assert all(info['actor_unchanged']) and sum(info['actor_updates'])==0
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)),np.arange(len(y)))
    stored=np.load(folder/'predictions.npz');prediction=stored['reward_incumbents']
    np.testing.assert_array_equal(merge(parts,best,len(y)),prediction[-1])
    held=np.ones(len(y),dtype=bool);held[supports]=False
    assert float(nmi(y[held],prediction[-1][held]))==result['heldout_nmi30']
    assert float(ari(y[held],prediction[-1][held]))==result['heldout_ari30']
    assert float(nmi(y[held],stored['full_training_incumbent'][held]))==result['full_training_heldout_nmi']
    np.testing.assert_allclose(np.maximum.accumulate([nmi(y,p) for p in prediction]),result['curve'],rtol=0,atol=1e-15)
    assert total_queries==result['total_queries'] and total_cuts==result['actual_cuts']
    assert result['clusters']>=len(parts)
    return dict(reward_checks=reward_checks,cut_checks=cut_checks,sweep_checks=sweep_checks)


def verify(out,source,workers):
    from concurrent.futures import ProcessPoolExecutor
    paths=sorted((out/'cases').glob('*/*/result.json'));expected={(s,d,m,seed) for s in SPACES for d in DATASETS for m in methods_for(s) for seed in range(10)}
    present=set()
    for p in paths:
        r=json.loads(p.read_text());present.add((r['space'],r['dataset'],r['method'],r['seed']))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(verify_case,[(str(p),str(source),str(out)) for p in paths]))
    missing=sorted(expected-present);unexpected=sorted(present-expected)
    report=dict(status='passed' if not missing and not unexpected else 'incomplete',verified_cases=len(paths),
                independent_reward_checks=sum(r['reward_checks'] for r in results),independent_cut_checks=sum(r['cut_checks'] for r in results),
                support_sweep_comparisons=sum(r['sweep_checks'] for r in results),
                missing=missing,unexpected=unexpected)
    write_json(out/'validation/artifact_checks.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='missing'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--output',type=Path,default=ROOT/'results/ac_joint');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();verify(args.output.resolve(),args.source.resolve(),args.workers)

"""Independent post-run checks of saved rewards, masks, selection and budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

from .prepare import ROOT,DATASETS,write_json
from .experiment import merge,methods_for


def verify_case(path):
    result=json.loads(path.read_text())
    if result['status']!='ok': raise AssertionError(f'Failed case: {path}')
    out=path.parents[3];folder=path.parent
    frozen=np.load(out/'inputs'/result['dataset']/f'seed{result["seed"]}.npz')
    y=frozen['y'];supports=frozen['reward_ids']
    parts=[];best_frames=[];reward_checks=0;total_fits=0;unique_fits=0;all_ids=[]
    for file in sorted(folder.glob('candidates_region*.npz'),key=lambda p:int(p.stem.replace('candidates_region',''))):
        region=int(file.stem.replace('candidates_region',''))
        saved=np.load(file);ids=saved['ids'];mask=saved['support_local_ids']
        np.testing.assert_array_equal(mask,np.flatnonzero(np.isin(ids,supports)))
        labels=saved['labels'];rewards=saved['rewards'];events=json.loads((folder/f'events_region{region}.json').read_text())
        assert len(labels)==len(rewards)==len(events)>0
        scores={}
        for prediction,reward in zip(labels,rewards):
            key=prediction[mask].tobytes()
            if key not in scores:
                scores[key]=float(nmi(y[ids[mask]],prediction[mask])) if len(mask)>=2 else 0.
                reward_checks+=1
            if reward!=scores[key]: raise AssertionError(f'Reward mismatch: {folder},region{region}')
        # Strict greater-than selects the first reward tie, using only first 30 entries.
        selected=int(np.argmax(rewards[:30]))
        parts.append(ids);best_frames.append(labels[selected]);all_ids.extend(ids)
        fits=json.loads((folder/f'fits_region{region}.json').read_text())
        total_fits+=len(fits);unique_fits+=len(set((f['u'],f['min_samples']) for f in fits))
        if result['method'] in ['Rand','BO-TPE','Anneal','PSO','GA','DE','MA-Random']:
            assert len(events)==30
        if 'FrozenActor' in result['method']:
            info=result['regions'][region]
            assert all(info['actor_unchanged']) and sum(info['actor_updates'])==0
    np.testing.assert_array_equal(np.sort(all_ids),np.arange(len(y)))
    prediction=np.load(folder/'predictions.npz')['reward_incumbents']
    np.testing.assert_array_equal(merge(parts,best_frames,len(y)),prediction[-1])
    held=np.ones(len(y),dtype=bool);held[supports]=False
    assert float(nmi(y[held],prediction[-1][held]))==result['heldout_nmi30']
    assert float(ari(y[held],prediction[-1][held]))==result['heldout_ari30']
    assert total_fits==result['actual_fits'] and unique_fits==result['unique_fits']
    assert np.all(np.diff(result['curve'])>=0)
    raw_values=np.asarray([nmi(y,p) for p in prediction])
    np.testing.assert_allclose(np.maximum.accumulate(raw_values),result['curve'],rtol=0,atol=1e-15)
    return reward_checks


def verify(out,workers):
    from concurrent.futures import ProcessPoolExecutor
    expected={(s,d,m,seed) for s in ['E','H','H-fixed-E','H-weak','H-strong']
              for d in DATASETS for m in methods_for(s) for seed in range(10)}
    paths=sorted((out/'cases').glob('*/*/result.json'))
    present=set()
    for p in paths:
        r=json.loads(p.read_text());present.add((r['space'],r['dataset'],r['method'],r['seed']))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checks=sum(pool.map(verify_case,paths))
    missing=sorted(expected-present)
    report=dict(status='passed' if not missing else 'incomplete',verified_cases=len(paths),
                unique_reward_checks=checks,missing=missing,unexpected=sorted(present-expected))
    write_json(out/'validation/artifact_checks.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='missing'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();verify(args.output.resolve(),args.workers)

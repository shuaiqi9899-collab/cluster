"""Independent public-library and intrinsic CF checks for selected models."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import Birch
from sklearn.metrics import adjusted_rand_score as ari
from hyperbolic.geometry import ball_distance,frechet
from hyperbolic.prepare import ROOT,write_json
from .context import Bank,canonical
from .lifecycle import ACTIVE_BACKENDS


def audit(out):
    banks={};checked=set();counts=dict(hdbscan_public=0,birch_public=0,native_cf=0,
                                     native_leaves=0,independent_frechet=0,proved_shortcuts=0)
    for path in sorted(out.glob('*/cases/*/*/result.json')):
        if path.parts[-5] not in ACTIVE_BACKENDS:continue
        r=json.loads(path.read_text())
        if r['status']!='ok':continue
        bkey=(r['backend'],r['dataset'],r['space'],r['seed'])
        if bkey not in banks:
            banks.clear()  # Limit memory across data/seed groups.
            banks[bkey]=Bank(ROOT/'results/hyperbolic',ROOT/'results/ac_joint',out,*bkey[1:3],bkey[0],bkey[3])
        bank=banks[bkey]
        for region,(info,selection) in enumerate(zip(r['regions'],r['selected_parameters'])):
            key=info['ids_key'];ids=bank.ids(key);params=selection['parameters']
            order=bank.order[np.isin(bank.order,ids)]
            content=dict(version='density-cf-v1',params=params,ids=key)
            if bank.backend=='BIRCH':content['order']=hashlib.sha256(order.tobytes()).hexdigest()
            digest=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
            fingerprint=(bank.backend,bank.dataset,bank.space,digest)
            cache=bank.namespace/key/(digest+'.npz')
            with np.load(cache) as f:
                labels=f['labels'];meta=json.loads(str(f['meta']))
                with np.load(path.parent/f'candidates_region{region}.npz') as candidates:
                    np.testing.assert_array_equal(candidates['ids'],ids)
                    best=int(np.argmax(candidates['scores'][:30]))
                    np.testing.assert_array_equal(labels,candidates['labels'][best])
                if fingerprint in checked:continue
                checked.add(fingerprint)
                if bank.backend=='HDBSCAN':
                    if len(ids)<2:assert np.all(labels==-1);continue
                    reference=HDBSCAN(metric='precomputed',algorithm='generic',**params).fit_predict(bank.distances[np.ix_(ids,ids)])
                    assert np.array_equal(labels<0,reference<0) and ari(labels,reference)==1.
                    counts['hdbscan_public']+=1
                elif 'shortcut' in meta:
                    assert params['threshold']<bank.delta/np.sqrt(2*len(bank.points))
                    _,reference=np.unique(bank.points[ids],axis=0,return_inverse=True)
                    assert ari(labels,reference)==1.;counts['proved_shortcuts']+=1
                elif not bank.radius:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');model=Birch(n_clusters=None,**params).fit(bank.points[order])
                    assert ari(labels,model.predict(bank.points[ids]))==1.;counts['birch_public']+=1
                else:
                    centers=f['centers'];members=f['member_ids'];offsets=f['offsets'];sses=f['leaf_sse']
                    np.testing.assert_array_equal(np.sort(members),ids)
                    assert ari(labels,ball_distance(bank.points[ids],centers).argmin(axis=1))==1.
                    for i,center in enumerate(centers):
                        local=members[offsets[i]:offsets[i+1]]
                        sse=float(np.sum(ball_distance(center[None],bank.points[local])**2))
                        assert abs(sse-sses[i])<=1e-9*(1+sse)
                        assert np.sqrt(sse/len(local))<=params['threshold']+1e-10
                        counts['native_leaves']+=1
                        if len(local)>1 and i<8:
                            q,_,_=frechet(bank.points[local],tolerance=1e-8)
                            reference=float(np.sum(ball_distance(q[None],bank.points[local])**2))
                            assert abs(reference-sse)<1e-8*(1+sse)
                            counts['independent_frechet']+=1
                    counts['native_cf']+=1
    result=dict(status='passed',backend_scope=list(ACTIVE_BACKENDS),unique_selected_models=len(checked),**counts)
    write_json(out/'backend_audit.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/density_cf');audit(p.parse_args().output.resolve())

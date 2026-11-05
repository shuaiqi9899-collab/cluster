"""Replay every saved E candidate through the independent H^3 control metric."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from .boundary import control_edge
from .geometry import ball_distance
from .prepare import DATASETS,ROOT,write_json


def replay(out):
    units=[]
    for dataset in DATASETS:
        canonical=np.load(out/'inputs'/dataset/'canonical.npz')
        x=canonical['x']; center=x.mean(0);z=x-center
        a=1/np.linalg.norm(z,axis=1).max();v=a*z;s=np.sum(v*v,axis=1,keepdims=True)
        p=np.concatenate((2*v,s),axis=1)/(s+4)
        dh=ball_distance(p);np.fill_diagonal(dh,0)
        de=np.load(out/'geometry'/dataset/'r0'/'distances.npy',mmap_mode='r')
        for seed in range(10):
            frozen=np.load(out/'inputs'/dataset/f'seed{seed}.npz')
            order=frozen['sample_ids'];cache={}
            for folder in sorted((out/'cases/E').glob(f'{dataset}_*_seed{seed}')):
                if not (folder/'result.json').exists(): continue
                result=json.loads((folder/'result.json').read_text())
                if result['status']!='ok': continue
                probes=[]
                for file in sorted(folder.glob('candidates_region*.npz')):
                    region=int(file.stem.replace('candidates_region',''))
                    f=np.load(file);ids=f['ids'];original_ids=order[ids]
                    e_dist=np.asarray(de[np.ix_(original_ids,original_ids)])
                    h_dist=dh[np.ix_(original_ids,original_ids)]
                    events=json.loads((folder/f'events_region{region}.json').read_text())
                    for i,event in enumerate(events):
                        eps,m=event['parameters'].split('+');eps,m=float(eps),int(float(m))
                        key=(ids.tobytes(),eps,m)
                        if key not in cache:
                            eh=float(2*np.arcsinh(a*eps/2))
                            em=DBSCAN(eps=eps,min_samples=m,metric='precomputed').fit(e_dist)
                            hm=DBSCAN(eps=eh,min_samples=m,metric='precomputed').fit(h_dist)
                            be,bh=e_dist<=eps,h_dist<=eh
                            ii,jj=np.where(np.triu(be!=bh,1))
                            differences=[]
                            for j,k in zip(ii,jj):
                                exact_e,exact_h=control_edge(x[original_ids[j]],x[original_ids[k]],center,a,eps)
                                differences.append(dict(i=int(original_ids[j]),j=int(original_ids[k]),
                                                        euclidean_gap=float(e_dist[j,k]-eps),
                                                        hyperbolic_gap=float(h_dist[j,k]-eh),
                                                        exact_euclidean=exact_e,exact_hyperbolic=exact_h))
                                if abs(e_dist[j,k]-eps)>1e-12 or abs(h_dist[j,k]-eh)>1e-12:
                                    raise AssertionError('Non-boundary metric mismatch')
                            core_equal=np.array_equal(em.core_sample_indices_,hm.core_sample_indices_)
                            noise_equal=np.array_equal(em.labels_==-1,hm.labels_==-1)
                            pred_ari=float(adjusted_rand_score(em.labels_,hm.labels_))
                            cache[key]=dict(e_labels=em.labels_,raw_adjacency_equal=len(ii)==0,
                                            boundary_pairs=differences,core_equal=core_equal,
                                            noise_equal=noise_equal,ari=pred_ari)
                        record=cache[key]
                        if not np.array_equal(record['e_labels'],f['labels'][i]):
                            raise AssertionError(f'Saved Euclidean labels do not replay: {folder}, region{region}, candidate{i}')
                        probes.append(dict(region=region,candidate=i,eps=eps,min_samples=m,
                                           **{k:v for k,v in record.items() if k!='e_labels'}))
                full_equal=all(p['core_equal'] and p['noise_equal'] and p['ari']==1 for p in probes)
                item=dict(dataset=dataset,seed=seed,method=result['method'],candidates=len(probes),
                          raw_adjacency_equal=all(p['raw_adjacency_equal'] for p in probes),
                          native_clustering_equal=full_equal,
                          boundary_pair_occurrences=sum(len(p['boundary_pairs']) for p in probes))
                write_json(out/'validation/replay'/f'{folder.name}.json',dict(summary=item,probes=probes))
                units.append(item)
                print('Replay',folder.name,len(probes),'candidates',full_equal,flush=True)
    write_json(out/'validation/replay_summary.json',dict(units=len(units),
               candidates=sum(u['candidates'] for u in units),
               native_clustering_equal=all(u['native_clustering_equal'] for u in units),
               raw_adjacency_equal=all(u['raw_adjacency_equal'] for u in units),
               boundary_pair_occurrences=sum(u['boundary_pair_occurrences'] for u in units),results=units))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/hyperbolic')
    replay(p.parse_args().output.resolve())

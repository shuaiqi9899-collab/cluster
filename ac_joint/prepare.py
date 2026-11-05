"""Label-free canonical tree bank, shared over optimizers and shuffled seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np
from hyperbolic.prepare import ROOT,DATASETS,write_json
from .trees import LINKAGES,make_tree


def auxiliary_groups(uncertainty,threshold=.3):
    order=np.argsort(uncertainty,kind='stable')
    groups=np.empty(len(order),dtype=int);group=0
    for j,index in enumerate(order):
        if j and uncertainty[index]-uncertainty[order[j-1]]>threshold:group+=1
        groups[index]=group
    return groups


def geometry_root(source,dataset,space):
    radius={'E':0.,'H':.5,'H-weak':.2,'H-strong':.8}[space]
    return source/'geometry'/dataset/f'r{radius:g}',radius


def subset_key(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()[:24]


def prepare_group(task):
    source,out,dataset,space=task;source,out=Path(source),Path(out)
    base,radius=geometry_root(source,dataset,space)
    points=np.load(base/'points.npy');distances=np.load(base/'distances.npy',mmap_mode='r')
    n=len(points);folder=out/'banks'/dataset/space;folder.mkdir(parents=True,exist_ok=True)
    subsets={subset_key(np.arange(n)):np.arange(n)}
    seed_regions={};aux_checks=[]
    for kind in ([space,'E'] if space=='H' else [space]):
        for seed in range(10):
            old=np.load(source/'partitions'/dataset/f'{kind}_seed{seed}.npz')
            groups=auxiliary_groups(old['uncertainty'])
            if not np.array_equal(groups[:,None]==groups[None,:],old['groups'][:,None]==old['groups'][None,:]):
                raise AssertionError('Auxiliary single-link grouping is not equivalent to saved min_samples=1 DBSCAN')
            labels=groups[old['modules']]
            order=np.load(source/'inputs'/dataset/f'seed{seed}.npz')['sample_ids']
            parts=[np.sort(order[labels==label]) for label in np.unique(labels)]
            parts.sort(key=lambda ids:int(ids[0]))
            keys=[]
            for ids in parts:
                key=subset_key(ids);subsets[key]=ids;keys.append(key)
            seed_regions[f'{kind}_seed{seed}']=keys
            aux_checks.append(dict(partition=kind,seed=seed,equivalent=True,regions=len(parts)))
    digest=hashlib.sha256((ROOT/'ac_joint/trees.py').read_bytes()).hexdigest()
    version=digest[:12]
    trees={};scales={method:0. for method in LINKAGES};build_seconds={method:0. for method in LINKAGES}
    for key,ids in subsets.items():
        trees[key]={}
        for method in LINKAGES:
            path=folder/f'{key}_{method}_{version}.npz'
            meta_path=path.with_suffix('.json')
            if not path.exists() or not meta_path.exists():
                d=np.asarray(distances[np.ix_(ids,ids)])
                z,h,meta=make_tree(points[ids],d,method,radius>0)
                audit=meta.pop('merge_audit',np.empty((0,4)))
                np.savez_compressed(path,ids=ids,z=z,heights=h,merge_audit=audit)
                meta.update(tree_source_sha256=digest,canonical_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
                write_json(meta_path,meta)
                print(dataset,space,method,len(ids),f'{meta["seconds"]:.2f}s',flush=True)
            meta=json.loads(meta_path.read_text())
            scales[method]=max(scales[method],meta['root_height'])
            build_seconds[method]+=meta['seconds']
            trees[key][method]=str(path.relative_to(out))
    index=dict(dataset=dataset,space=space,radius=radius,n=n,version=version,source_sha256=digest,
               global_key=subset_key(np.arange(n)),regions=seed_regions,trees=trees,scales=scales,
               build_seconds=build_seconds,auxiliary_checks=aux_checks,
               scale_rule='Maximum effective root height over the global tree and all actual native/fixed-E region trees, shared across seeds/regions',
               ties='Canonical original sample ID order; linkage outputs deterministic under shuffled input order')
    write_json(folder/'index.json',index)
    return dict(dataset=dataset,space=space,unique_subsets=len(subsets),build_seconds=build_seconds)


def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic')
    parser.add_argument('--output',type=Path,default=ROOT/'results/ac_joint')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--spaces',nargs='+',choices=['E','H','H-weak','H-strong'],default=['E','H','H-weak','H-strong'])
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    tasks=[(str(args.source.resolve()),str(out),d,s) for d in args.datasets for s in args.spaces]
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(prepare_group,t) for t in tasks]):summaries.append(future.result())
    write_json(out/'preparation.json',dict(status='passed',groups=summaries,source=str(args.source.resolve()),
                                         linkages=LINKAGES,ward='Native Fréchet-SSE, Newton residual <=1e-9; conservative numerical upper ordering',
                                         source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'ac_joint').glob('*.py')}))
    print('All selected tree banks prepared.',flush=True)


if __name__=='__main__':main()

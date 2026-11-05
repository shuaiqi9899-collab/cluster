"""Make lightweight region-ID banks; no AC/Ward tree construction is needed."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from hyperbolic.prepare import ROOT,DATASETS,write_json


def key(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()[:24]


def prepare(source,out):
    groups=[]
    for dataset in DATASETS:
        for space,radius in [('E',0.),('H',.5),('H-weak',.2),('H-strong',.8)]:
            n=len(np.load(source/'geometry'/dataset/f'r{radius:g}'/'points.npy',mmap_mode='r'))
            allids=np.arange(n,dtype=np.int64);subsets={key(allids):allids};regions={}
            for kind in ([space,'E'] if space=='H' else [space]):
                for seed in range(10):
                    partition=np.load(source/'partitions'/dataset/f'{kind}_seed{seed}.npz')['regions']
                    order=np.load(source/'inputs'/dataset/f'seed{seed}.npz')['sample_ids']
                    parts=[np.sort(order[partition==label]) for label in np.unique(partition)]
                    parts.sort(key=lambda ids:int(ids[0]));keys=[]
                    for ids in parts:subsets[key(ids)]=ids;keys.append(key(ids))
                    regions[f'{kind}_seed{seed}']=keys
            folder=out/'banks'/dataset/space;folder.mkdir(parents=True,exist_ok=True);trees={}
            for digest,ids in subsets.items():
                path=folder/f'{digest}_ids.npz';np.savez_compressed(path,ids=ids)
                trees[digest]={'single':str(path.relative_to(out))}
            # `trees/single` is only a backwards-compatible ID-bank adapter key.
            # Files contain IDs, not linkage trees or labels.
            write_json(folder/'index.json',dict(dataset=dataset,space=space,radius=radius,n=n,
                                                global_key=key(allids),regions=regions,trees=trees,
                                                format='region-ids-only-v1; no clustering trees'))
            groups.append(dict(dataset=dataset,space=space,subsets=len(subsets)))
    write_json(out/'preparation.json',dict(status='passed',groups=groups,source=str(source)))
    return groups


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic')
    p.add_argument('--output',type=Path,default=ROOT/'results/density_cf/region_inputs')
    a=p.parse_args();print('Prepared',len(prepare(a.source.resolve(),a.output.resolve())),'lightweight region banks')

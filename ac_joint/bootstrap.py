"""Prepare missing geometry/SE modules without running DBSCAN training."""
import argparse
from pathlib import Path
import time

import numpy as np
import hyperbolic.experiment as previous  # initialize dependencies before forking
from hyperbolic.prepare import ROOT,DATASETS,prepare as prepare_geometry
from .prepare import auxiliary_groups


def make_partition(task):
    source,dataset,space,seed=task;source=Path(source)
    path=source/'partitions'/dataset/f'{space}_seed{seed}.npz'
    if path.exists():return 'reused'
    geometry,_=previous.load_geometry(source,dataset,seed,space)
    original=previous.DBSCAN
    class ScalarGroups:
        def __init__(self,eps,min_samples):
            if eps!=.3 or min_samples!=1:raise ValueError('Only auxiliary scalar grouping is permitted')
        def fit_predict(self,x):return auxiliary_groups(np.asarray(x).reshape(-1))
    start=time.perf_counter()
    try:
        previous.DBSCAN=ScalarGroups
        result=previous.partition(geometry,seed)
    finally:previous.DBSCAN=original
    result['preparation_seconds']=time.perf_counter()-start
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result)
    print('Prepared SE',space,dataset,seed,flush=True)
    return 'prepared'


def main():
    from concurrent.futures import ProcessPoolExecutor
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();source=args.source.resolve()
    if not (source/'validation/geometry_gate.json').exists():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError('Incomplete nonempty source: preserve it and choose an empty --source directory')
        prepare_geometry(source)
    tasks=[(str(source),d,s,seed) for d in DATASETS for s in ['E','H','H-weak','H-strong'] for seed in range(10)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(make_partition,tasks))
    print('SE sources:',results.count('prepared'),'prepared,',results.count('reused'),'reused; no DBSCAN training.',flush=True)


if __name__=='__main__':main()

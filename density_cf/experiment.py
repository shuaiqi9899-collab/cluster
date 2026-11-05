"""HDBSCAN-only public entry point; original training engine remains frozen."""
import argparse
from pathlib import Path
import sys
from hyperbolic.prepare import ROOT,DATASETS
from . import _frozen_experiment_v1 as engine
from .lifecycle import ACTIVE_BACKENDS,require_active

METHODS=engine.METHODS
SPACES=engine.SPACES
methods_for=engine.methods_for
merge=engine.merge


def run_case(source,out,bank,dataset,space,seed,method):
    require_active(bank.backend)
    return engine.run_case(source,out,bank,dataset,space,seed,method)


def group_worker(task):
    require_active(task[3])
    return engine.group_worker(task)


def main():
    p=argparse.ArgumentParser(description='Active backend: HDBSCAN. BIRCH is retired by user instruction.')
    p.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic')
    p.add_argument('--output',type=Path,default=ROOT/'results/density_cf')
    p.add_argument('--ac-source',type=Path,default=ROOT/'results/ac_joint')
    p.add_argument('--backends',nargs='+',choices=ACTIVE_BACKENDS,default=list(ACTIVE_BACKENDS))
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--spaces',nargs='+',choices=SPACES,default=SPACES)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--methods',nargs='+',choices=METHODS)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    arguments=['--source',str(a.source),'--output',str(a.output),'--ac-source',str(a.ac_source),
               '--backends',*a.backends,'--datasets',*a.datasets,'--spaces',*a.spaces,
               '--seeds',*[str(s) for s in a.seeds],'--workers',str(a.workers)]
    if a.methods is not None:arguments+=['--methods',*a.methods]
    original=sys.argv
    try:
        sys.argv=[original[0]]+arguments
        engine.main()
    finally:sys.argv=original


if __name__=='__main__':main()

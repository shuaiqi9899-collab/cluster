"""Pinned HDBSCAN hierarchy/extraction, independent of support labels."""
import time
import numpy as np
from hdbscan.hdbscan_ import _hdbscan_generic,_tree_to_labels

MIN_SAMPLES=(1,5,10,20)


def build_hierarchy(distances,min_samples):
    start=time.perf_counter();n=len(distances)
    actual=max(1,min(int(min_samples),n-1))
    if n<2:return np.empty((0,4)),dict(seconds=0.,min_samples=actual,n=n)
    tree,_=_hdbscan_generic(np.asarray(distances,dtype=np.float64),min_samples=actual,
                           alpha=1.,metric='precomputed',gen_min_span_tree=False)
    return tree,dict(seconds=time.perf_counter()-start,min_samples=actual,n=n,
                     convention='hdbscan 0.8.40 generic precomputed; min_samples excludes self',
                     cluster_selection='eom',allow_single_cluster=False,alpha=1.)


def extract(tree,n,min_cluster_size):
    if n<2:return np.full(n,-1,dtype=int),np.zeros(n),dict(clusters=0,seconds=0.)
    start=time.perf_counter()
    labels,probabilities,persistence,condensed,_=_tree_to_labels(
        None,tree,min_cluster_size=int(min_cluster_size),cluster_selection_method='eom',
        allow_single_cluster=False,cluster_selection_epsilon=0.,cluster_selection_persistence=0.,
        match_reference_implementation=False,max_cluster_size=0)
    return labels,probabilities,dict(clusters=len(set(labels)-{-1}),seconds=time.perf_counter()-start,
                                   condensed_rows=len(condensed))

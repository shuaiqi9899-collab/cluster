"""Label-free clustering bank plus support-only search adapter.

The inherited optimizer uses `linkage_id` internally for its four categorical
actions. Public records name the actual backend parameters instead.
"""
import collections
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import normalized_mutual_info_score as nmi

from ac_joint.context import GeometryBank
from hyperbolic.prepare import write_json
from .hdb import MIN_SAMPLES,build_hierarchy,extract
from .birch import BRANCHING,Statistics,GeodesicBirch


def canonical(labels):
    labels=np.asarray(labels,dtype=int);result=np.full(len(labels),-1,dtype=int)
    nonnoise=labels>=0
    if np.any(nonnoise):
        _,first,inverse=np.unique(labels[nonnoise],return_index=True,return_inverse=True)
        rank=np.empty(len(first),dtype=int);rank[np.argsort(first)]=np.arange(len(first))
        result[nonnoise]=rank[inverse]
    return result


class Bank(GeometryBank):
    def __init__(self,source,ac_source,out,dataset,space,backend,seed):
        super().__init__(source,ac_source,dataset,space)
        self.output=Path(out);self.backend=backend;self.seed=seed
        base=Path(source)/'geometry'/dataset/f'r{self.radius:g}'
        self.distances=np.load(base/'distances.npy',mmap_mode='r')
        self.order=np.load(Path(source)/'inputs'/dataset/f'seed{seed}.npz')['sample_ids']
        positive=self.distances[self.distances>0]
        self.delta=float(positive.min()) if len(positive) else 1.
        self.tau_min=self.delta/(10*np.sqrt(len(self.points)))
        self.tau_max=max(self.tau_min,float(self.distances.max()))
        self.stats=Statistics(self.points,hyperbolic=bool(self.radius),cache_size=50000)
        self.memory={};self.hierarchies={};self.ids_cache={}
        self.category_values=MIN_SAMPLES if backend=='HDBSCAN' else BRANCHING
        self.category_name='min_samples' if backend=='HDBSCAN' else 'branching_factor'
        self.namespace=self.output/'cache'/backend/dataset/self.space
    def ids(self,key):
        if key not in self.ids_cache:
            # Reuse frozen, label-free region membership; do not load AC trees.
            path=self.out/self.index['trees'][key]['single']
            self.ids_cache[key]=np.load(path)['ids']
        return self.ids_cache[key]
    def parameters(self,key,category,u):
        n=len(self.ids(key));category=int(category);u=float(np.clip(u,0,1))
        if self.backend=='HDBSCAN':
            maximum=max(2,int(np.ceil(n/4)))
            size=int(np.clip(np.rint(np.exp(np.log(2)+u*np.log(maximum/2))),2,maximum))
            return dict(min_samples=max(1,min(MIN_SAMPLES[category],n-1)),min_cluster_size=size)
        return dict(branching_factor=BRANCHING[category],
                    threshold=float(np.exp(np.log(self.tau_min)+u*np.log(self.tau_max/self.tau_min))))
    def predict(self,key,params):
        ids=self.ids(key);order=self.order[np.isin(self.order,ids)]
        version='density-cf-v1'
        content=dict(version=version,params=params,ids=key)
        if self.backend=='BIRCH':content['order']=hashlib.sha256(order.tobytes()).hexdigest()
        digest=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
        file=self.namespace/key/(digest+'.npz')
        if digest in self.memory:
            labels,meta=self.memory[digest];return labels,dict(meta,cache_hit=True,cache_source='memory')
        if file.exists():
            with np.load(file) as f:labels=f['labels'];meta=json.loads(str(f['meta']))
            self.memory[digest]=(labels,meta)
            return labels,dict(meta,cache_hit=True,cache_source='disk')
        start=time.perf_counter();extra={}
        if self.backend=='HDBSCAN':
            k=params['min_samples'];hfile=self.namespace/key/f'hierarchy_ms{k}.npz'
            hierarchy_hit=(key,k) in self.hierarchies or hfile.exists()
            if (key,k) not in self.hierarchies:
                if hfile.exists():
                    with np.load(hfile) as f:tree=f['tree'];hmeta=json.loads(str(f['meta']))
                else:
                    tree,hmeta=build_hierarchy(self.distances[np.ix_(ids,ids)],k)
                    hfile.parent.mkdir(parents=True,exist_ok=True)
                    # Workers may share a tree: atomic replacement, same deterministic content.
                    self.save(hfile,tree=tree,meta=json.dumps(hmeta))
                self.hierarchies[key,k]=(tree,hmeta)
            tree,hmeta=self.hierarchies[key,k]
            labels,probabilities,meta=extract(tree,len(ids),params['min_cluster_size'])
            extra['probabilities']=probabilities
            meta.update(hierarchy_cache_hit=hierarchy_hit,hierarchy_seconds=hmeta['seconds'],
                        hierarchy_new_seconds=0. if hierarchy_hit else hmeta['seconds'])
        else:
            tau=params['threshold'];branch=params['branching_factor']
            # CAT(0) variance inequality: any cluster containing a distinct pair
            # has SSE >= delta^2/2, hence RMS >= delta/sqrt(2*N).
            # Below this bound only identical points can merge, for every stream.
            if tau<self.delta/np.sqrt(2*len(self.points)):
                _,labels=np.unique(self.points[ids],axis=0,return_inverse=True)
                meta=dict(shortcut='proved distinct-pair RMS lower bound',splits=None,
                          leaf_subclusters=len(np.unique(labels)),frechet_solves=0,mean_cache_hits=0)
            elif self.radius:
                model=GeodesicBirch(self.stats,tau,branch);labels,meta=model.fit_predict(order)
                extra.update(centers=model.centers,leaf_sse=[e.sse for e in model.leaf_entries],
                             member_ids=np.concatenate([e.ids for e in model.leaf_entries]),
                             offsets=np.r_[0,np.cumsum([len(e.ids) for e in model.leaf_entries])])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');model=Birch(threshold=tau,branching_factor=branch,n_clusters=None).fit(self.points[order])
                labels=model.predict(self.points[ids]);extra['centers']=model.subcluster_centers_
                meta=dict(leaf_subclusters=len(model.subcluster_centers_),implementation='sklearn 1.2.2 Birch n_clusters=None')
        labels=canonical(labels)
        meta.update(seconds=time.perf_counter()-start,parameters=params,backend=self.backend)
        self.save(file,labels=labels,meta=json.dumps(meta),**extra)
        self.memory[digest]=(labels,meta)
        return labels,dict(meta,cache_hit=False,cache_source='new')
    @staticmethod
    def save(path,**values):
        import os
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp,**values);temp.replace(path)


class Context:
    def __init__(self,bank,key,support_ids,support_labels):
        self.bank=bank;self.key=key;self.ids=bank.ids(key)
        self.mask=np.flatnonzero(np.isin(self.ids,support_ids))
        self.support_labels=np.asarray(support_labels).copy()
        if len(self.mask)!=len(self.support_labels):raise ValueError('Support ordering mismatch')
        self.events=[];self.queries=[];self.state_cache={};self.partitions={}
        self.costs=collections.Counter();self.distinct_pairs=set();self.distinct_predictions=set();self.physical_pairs=set()
    def query(self,linkage,u,mode='train'):
        start=time.perf_counter();category=int(linkage);u=float(np.clip(u,0,1))
        self.costs[f'{mode}_requests']+=1
        params=self.bank.parameters(self.key,category,u);pair=json.dumps(params,sort_keys=True)
        if pair in self.partitions:
            labels,meta=self.partitions[pair];meta=dict(meta,cache_hit=True,cache_source='context')
        else:
            labels,meta=self.bank.predict(self.key,params);self.partitions[pair]=(labels,meta)
        if meta['cache_hit']:self.costs[f'{mode}_cache_hits']+=1
        else:
            self.costs['actual_fits']+=1;self.costs['fit_seconds']+=meta['seconds']
            self.costs['hierarchy_new_seconds']+=meta.get('hierarchy_new_seconds',0.)
        self.costs['cut_query_seconds']+=time.perf_counter()-start
        self.distinct_pairs.add((category,u));self.physical_pairs.add(pair);self.distinct_predictions.add(labels.tobytes())
        event=dict(category_id=category,category_value=self.bank.category_values[category],u=u,
                   parameters=params,mode=mode,cache_hit=meta['cache_hit'],cache_source=meta['cache_source'],
                   seconds=time.perf_counter()-start,actual_fits=self.costs['actual_fits'],
                   cumulative_queries=len(self.queries)+1,clusters=len(set(labels)-{-1}),noise_fraction=float(np.mean(labels<0)))
        self.queries.append(event.copy())
        if mode=='train':
            event.update(linkage_id=category,score=float(nmi(self.support_labels,labels[self.mask])) if len(self.mask)>=2 else 0.,labels=labels)
            self.events.append(event)
        return labels
    def score(self):return self.events[-1]['score']
    def state(self,labels,linkage,u,bounds):
        key=labels.tobytes()
        if key not in self.state_cache:
            center=self.bank.mean(self.ids);local=[]
            # A noise token retains the same four-dimensional state architecture.
            # Noise is pooled for state/reward, never converted into a cluster.
            for label in np.unique(labels):
                ids=self.ids[labels==label];q=self.bank.mean(ids)
                local.append([self.bank.distance(q,center)/self.bank.mapping['scale'],len(ids)/100]+list(self.bank.chart(q)))
            self.state_cache[key]=local
        k=len(set(labels)-{-1})
        global_state=[u,u-bounds[0],bounds[1]-u,k/len(labels)]+[float(i==linkage) for i in range(4)]
        return [[global_state],self.state_cache[key]]

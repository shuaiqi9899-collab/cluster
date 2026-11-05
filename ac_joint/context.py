"""Only permitted support labels enter this search interface."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from hyperbolic.geometry import ball_distance,logmap
from .trees import LINKAGES,cut_tree,mean_sse


class GeometryBank:
    def __init__(self,source,out,dataset,space):
        self.source=Path(source);self.out=Path(out);self.dataset=dataset
        self.space='H' if space=='H-fixed-E' else space
        self.index=json.loads((self.out/'banks'/dataset/self.space/'index.json').read_text())
        radius=self.index['radius'];base=self.source/'geometry'/dataset/f'r{radius:g}'
        self.radius=radius;self.points=np.load(base/'points.npy')
        self.mapping=json.loads((base/'mapping.json').read_text())
        self.trees={};self.center_cache={};self.mean_calls=0;self.max_residual=0.
    def load(self,key):
        if key not in self.trees:
            values=[]
            for method in LINKAGES:
                f=np.load(self.out/self.index['trees'][key][method])
                values.append((f['z'],f['heights'],f['ids']))
            self.trees[key]=values
        return self.trees[key]
    def mean(self,ids):
        ids=np.asarray(ids,dtype=int);key=ids.tobytes()
        if key not in self.center_cache:
            if len(ids)==1:q=self.points[ids[0]]
            elif self.radius:
                q,_,res,_=mean_sse(self.points[ids]);self.mean_calls+=1
                self.max_residual=max(self.max_residual,res)
            else:q=self.points[ids].mean(axis=0)
            self.center_cache[key]=q
        return self.center_cache[key]
    def chart(self,q):
        return np.asarray(self.mapping['center'])+logmap(np.zeros(2),q)/self.mapping['a'] if self.radius else q
    def distance(self,p,q):
        return float(ball_distance(p[None],q[None])[0,0]) if self.radius else float(np.linalg.norm(p-q))


class Context:
    def __init__(self,bank,key,support_ids,support_labels):
        self.bank=bank;self.key=key;self.trees=bank.load(key);self.ids=self.trees[0][2]
        self.mask=np.flatnonzero(np.isin(self.ids,support_ids))
        if len(self.mask)!=len(support_labels):raise ValueError('Support labels must match the canonical local support order')
        self.support_labels=np.asarray(support_labels).copy()
        self.events=[];self.queries=[];self.partitions={};self.state_cache={}
        self.costs=collections.Counter();self.distinct_pairs=set();self.distinct_predictions=set()
    def query(self,linkage,u,mode='train'):
        start=time.perf_counter();linkage=int(linkage);u=float(np.clip(u,0,1));pair=(linkage,u)
        self.costs[f'{mode}_requests']+=1;cached=pair in self.partitions
        if not cached:
            z,h,_=self.trees[linkage]
            threshold=u*self.bank.index['scales'][LINKAGES[linkage]]
            labels=cut_tree(z,h,len(self.ids),threshold)
            _,first,inverse=np.unique(labels,return_index=True,return_inverse=True)
            rank=np.empty(len(first),dtype=int);rank[np.argsort(first)]=np.arange(len(first))
            labels=rank[inverse]
            self.partitions[pair]=labels
            self.costs['actual_cuts']+=1
        else:self.costs[f'{mode}_cache_hits']+=1
        labels=self.partitions[pair]
        self.distinct_pairs.add(pair);self.distinct_predictions.add(labels.tobytes())
        elapsed=time.perf_counter()-start;self.costs['cut_query_seconds']+=elapsed
        self.queries.append(dict(linkage=LINKAGES[linkage],u=u,
                                 threshold=u*self.bank.index['scales'][LINKAGES[linkage]],
                                 mode=mode,cache_hit=cached,seconds=elapsed,actual_cuts=self.costs['actual_cuts']))
        if mode=='train':
            score=float(nmi(self.support_labels,labels[self.mask])) if len(self.mask)>=2 else 0.
            self.events.append(dict(linkage=LINKAGES[linkage],linkage_id=linkage,u=u,
                                   threshold=u*self.bank.index['scales'][LINKAGES[linkage]],
                                   score=score,labels=labels,actual_cuts=self.costs['actual_cuts'],
                                   cumulative_queries=len(self.queries),cache_hit=cached))
        return labels
    def score(self):return self.events[-1]['score']
    def state(self,labels,linkage,u,bounds):
        key=labels.tobytes()
        if key not in self.state_cache:
            center=self.bank.mean(self.ids);local=[]
            for label in range(int(labels.max())+1):
                ids=self.ids[labels==label];q=self.bank.mean(ids)
                local.append([self.bank.distance(q,center)/self.bank.mapping['scale'],len(ids)/100]+list(self.bank.chart(q)))
            self.state_cache[key]=local
        onehot=[float(i==linkage) for i in range(4)]
        global_state=[u,u-bounds[0],bounds[1]-u,(int(labels.max())+1)/len(labels)]+onehot
        return [[global_state],self.state_cache[key]]


def warmup(ctx,seed):
    for linkage in np.roll(np.arange(4),seed%4):ctx.query(int(linkage),.5)
    index=int(np.argmax([e['score'] for e in ctx.events]))
    return ctx.events[index]

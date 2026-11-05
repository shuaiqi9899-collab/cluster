"""CF-tree extension using exact-objective Fréchet means and geodesic RMS radii.

Unlike Euclidean CF (N,LS,SS), members are retained: intrinsic Fréchet statistics
are not an additive finite-dimensional sufficient statistic. This does not claim
the original BIRCH streaming-memory bound.
"""
import collections
import hashlib
import time

import numpy as np
from scipy.spatial.distance import cdist
from hyperbolic.geometry import ball_distance,expmap,logmap
from ac_joint.trees import mean_sse

BRANCHING=(10,25,50,100)


class Statistics:
    def __init__(self,points,hyperbolic=True,cache_size=50000):
        self.points=np.asarray(points,dtype=np.float64);self.hyperbolic=hyperbolic
        self.cache=collections.OrderedDict();self.cache_size=cache_size
        self.calls=self.hits=0;self.max_residual=0.
    def distances(self,x,y=None):
        return ball_distance(x,y) if self.hyperbolic else cdist(x,x if y is None else y)
    def summarize(self,ids,warm=None):
        ids=np.sort(np.asarray(ids,dtype=np.int64));key=hashlib.sha256(ids.tobytes()).digest()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return ids,*self.cache[key]
        if len(ids)==1:q=self.points[ids[0]];sse=res=0.
        elif not self.hyperbolic:
            q=self.points[ids].mean(axis=0);sse=float(np.sum((self.points[ids]-q)**2));res=0.
        else:q,sse,res,_=mean_sse(self.points[ids],warm);self.calls+=1;self.max_residual=max(self.max_residual,res)
        self.cache[key]=(q,sse,res)
        if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return ids,q,sse,res


class Entry:
    def __init__(self,stats,ids,child=None,warm=None):
        self.ids,self.center,self.sse,self.residual=stats.summarize(ids,warm)
        self.child=child
    @property
    def radius2(self):return self.sse/len(self.ids)


class Node:
    def __init__(self,leaf=True):
        self.leaf=leaf;self.entries=[];self.prev=None;self.next=None


class GeodesicBirch:
    def __init__(self,stats,threshold,branching_factor):
        if threshold<=0 or branching_factor<2:raise ValueError('Positive radius and branching_factor>=2 required')
        self.stats=stats;self.threshold=float(threshold);self.branching_factor=int(branching_factor)
        self.dummy=Node();self.root=Node();self.dummy.next=self.root;self.root.prev=self.dummy
        self.splits=0;self.boundary_checks=0;self.max_leaf_entries=0
    def summary(self,node):
        ids=np.concatenate([entry.ids for entry in node.entries])
        return Entry(self.stats,ids,child=node)
    def split(self,node):
        self.splits+=1;centers=np.array([e.center for e in node.entries])
        distance=self.stats.distances(centers)
        a,b=np.unravel_index(np.argmax(distance),distance.shape)
        closer=distance[a]<distance[b];closer[a]=True
        left,right=Node(node.leaf),Node(node.leaf)
        left.entries=[e for i,e in enumerate(node.entries) if closer[i]]
        right.entries=[e for i,e in enumerate(node.entries) if not closer[i]]
        if not left.entries or not right.entries:raise RuntimeError('Degenerate CF split')
        if node.leaf:
            left.prev=node.prev;left.next=right;right.prev=left;right.next=node.next
            if node.prev is not None:node.prev.next=left
            if node.next is not None:node.next.prev=right
        return self.summary(left),self.summary(right)
    def combine(self,first,second):
        warm=None
        if self.stats.hyperbolic:
            fraction=len(second.ids)/(len(first.ids)+len(second.ids))
            warm=expmap(first.center,fraction*logmap(first.center,second.center))
        return Entry(self.stats,np.r_[first.ids,second.ids],first.child,warm)
    def insert(self,node,point):
        if not node.entries:node.entries.append(point);return False
        centers=np.array([e.center for e in node.entries])
        index=int(np.argmin(self.stats.distances(centers,point.center[None])[:,0]));closest=node.entries[index]
        if closest.child is not None:
            split=self.insert(closest.child,point)
            if not split:node.entries[index]=self.combine(closest,point)
            else:
                a,b=self.split(closest.child);node.entries[index]=a;node.entries.append(b)
        else:
            candidate=self.combine(closest,point)
            gap=candidate.radius2-self.threshold**2
            numerical=(candidate.residual**2+128*np.finfo(float).eps*(1+candidate.sse)/len(candidate.ids))
            if abs(gap)<=numerical:
                # Record ambiguous float64 boundary decisions; do not widen threshold.
                self.boundary_checks+=1
            if candidate.radius2<=self.threshold**2:node.entries[index]=candidate
            else:node.entries.append(point)
            self.max_leaf_entries=max(self.max_leaf_entries,len(node.entries))
        return len(node.entries)>self.branching_factor
    def fit_predict(self,order):
        start=time.perf_counter();calls=self.stats.calls;hits=self.stats.hits
        for index in order:
            point=Entry(self.stats,[index]);split=self.insert(self.root,point)
            if split:
                a,b=self.split(self.root);self.root=Node(False);self.root.entries=[a,b]
        entries=[];leaf=self.dummy.next
        while leaf is not None:entries.extend(leaf.entries);leaf=leaf.next
        centers=np.array([e.center for e in entries])
        self.leaf_entries=entries
        self.centers=centers
        labels=np.argmin(self.stats.distances(self.stats.points[np.sort(order)],centers),axis=1)
        radius=np.sqrt([entry.radius2 for entry in entries])
        if np.any(radius>self.threshold*(1+1e-10)+1e-12):raise RuntimeError('Leaf CF radius exceeds threshold')
        return labels,dict(seconds=time.perf_counter()-start,splits=self.splits,leaf_subclusters=len(entries),
                           max_leaf_entries=self.max_leaf_entries,frechet_solves=self.stats.calls-calls,
                           mean_cache_hits=self.stats.hits-hits,max_mean_residual=self.stats.max_residual,
                           float_boundary_checks=self.boundary_checks,
                           max_leaf_radius=float(radius.max()),member_ids_retained=True,
                           final_assignment='Nearest leaf center in native metric, as BIRCH n_clusters=None')

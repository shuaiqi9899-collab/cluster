"""Intrinsic H-only clustering and support-only optimization interface."""
import collections,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from hyperbolic.geometry import ball_distance,logmap
from ac_joint.trees import LINKAGES,cut_tree,mean_sse
from density_cf.hdb import MIN_SAMPLES,build_hierarchy,extract
from .common import write,sha


def canonical(labels):
 labels=np.asarray(labels,dtype=int);result=np.full(len(labels),-1,dtype=int);valid=labels>=0
 if np.any(valid):
  _,first,inverse=np.unique(labels[valid],return_index=True,return_inverse=True)
  rank=np.empty(len(first),dtype=int);rank[np.argsort(first)]=np.arange(len(first));result[valid]=rank[inverse]
 return result


class Bank:
 def __init__(self,out,key,backend):
  if backend not in ['DBSCAN','AC','HDBSCAN']:raise ValueError('Only the three authorized backends are permitted')
  self.out=Path(out);self.key=key;self.backend=backend;self.folder=self.out/'prepared'/key
  self.index=json.loads((self.folder/'preparation.json').read_text());self.meta=json.loads((self.folder/'geometry.json').read_text())
  if self.meta['space']!='H' or self.meta['euclidean_clustering']:raise ValueError('H-only gate failed')
  self.geometry_sha256=sha(self.folder/'geometry.json')
  self.points=np.load(self.folder/'points.npy');self.distance=np.load(self.folder/'distances.npy',mmap_mode='r')
  self.subsets={};self.trees={};self.local_distances=collections.OrderedDict();self.centers=collections.OrderedDict();self.hierarchies={}
  self.mean_calls=0;self.max_residual=0.;self.predictions={}
 def ids(self,key):
  if key not in self.subsets:self.subsets[key]=np.load(self.folder/'subsets'/f'{key}_ids.npy')
  return self.subsets[key]
 def distances(self,key):
  if key==self.index['global_key']:return self.distance
  if key not in self.local_distances:
   ids=self.ids(key);self.local_distances[key]=np.asarray(self.distance[np.ix_(ids,ids)])
   if len(self.local_distances)>4:self.local_distances.popitem(last=False)
  self.local_distances.move_to_end(key);return self.local_distances[key]
 def mean(self,ids):
  key=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).digest()
  if key not in self.centers:
   if len(ids)==1:q=self.points[ids[0]]
   else:
    q,_,res,_=mean_sse(self.points[ids]);self.mean_calls+=1;self.max_residual=max(self.max_residual,res)
   self.centers[key]=q
   if len(self.centers)>40000:self.centers.popitem(last=False)
  self.centers.move_to_end(key);return self.centers[key]
 def chart(self,q):
  # Only the policy's two position features use this fixed projection. All
  # clustering/SE distances and Frechet objectives retain the full dimension.
  tangent=logmap(np.zeros_like(q),q)/self.meta['a']
  return tangent@np.asarray(self.meta['state_projection'])/max(np.sqrt(self.points.shape[1]),1.)
 def params(self,key,a,u):
  n=len(self.ids(key));maximum=max(1,int(np.ceil(n/4)))
  if self.backend=='DBSCAN':
   lower=self.meta['eps_min'];upper=self.meta['eps_max']
   return dict(eps=float(np.exp(np.log(lower)+float(a)*np.log(upper/lower))),min_samples=int(np.clip(np.rint(1+u*(maximum-1)),1,maximum)))
  if self.backend=='AC':return dict(linkage=LINKAGES[int(a)],threshold=float(u*self.index['scales'][LINKAGES[int(a)]]))
  return dict(min_samples=max(1,min(MIN_SAMPLES[int(a)],n-1)),min_cluster_size=int(np.clip(np.rint(np.exp(np.log(2)+u*np.log(max(2,maximum)/2))),2,max(2,maximum))))
 def predict(self,key,params):
  content=dict(version='benchmarks80-H-v1',geometry_sha=self.geometry_sha256,data_sha=self.index['data_sha256'],subset=key,backend=self.backend,parameters=params)
  digest=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
  file=self.out/'cache'/self.key/self.backend/key/(digest+'.npz')
  if digest in self.predictions:return self.predictions[digest],dict(cache_hit=True,source='memory',seconds=0.)
  if file.exists():
   with np.load(file) as f:labels=f['labels']
   self.predictions[digest]=labels;return labels,dict(cache_hit=True,source='disk',seconds=0.)
  start=time.perf_counter();ids=self.ids(key);extra={}
  if self.backend=='DBSCAN':labels=DBSCAN(metric='precomputed',**params).fit_predict(self.distances(key))
  elif self.backend=='AC':
   method=params['linkage'];tkey=(key,method)
   if tkey not in self.trees:
    with np.load(self.folder/self.index['trees'][key][method]) as f:self.trees[tkey]=(f['z'],f['heights'])
   z,h=self.trees[tkey];labels=cut_tree(z,h,len(ids),params['threshold'])
  else:
   k=params['min_samples'];hkey=(key,k);hfile=self.out/'cache'/self.key/'HDBSCAN'/key/f'hierarchy_ms{k}.npz'
   if hkey not in self.hierarchies:
    if hfile.exists():
     with np.load(hfile) as f:tree=f['tree'];hmeta=json.loads(str(f['meta']))
    else:
     tree,hmeta=build_hierarchy(self.distances(key),k);self.save(hfile,tree=tree,meta=json.dumps(hmeta))
    self.hierarchies[hkey]=(tree,hmeta)
   tree,hmeta=self.hierarchies[hkey];labels,_,extra=extract(tree,len(ids),params['min_cluster_size'])
  labels=canonical(labels);meta=dict(extra);meta.update(cache_hit=False,source='new',seconds=time.perf_counter()-start)
  self.save(file,labels=labels,meta=json.dumps(meta),parameters=json.dumps(params))
  self.predictions[digest]=labels;return labels,meta
 @staticmethod
 def save(path,**values):
  import os
  path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz');np.savez_compressed(temp,**values);temp.replace(path)


class Context:
 def __init__(self,bank,key,support_ids,support_labels):
  self.bank=bank;self.key=key;self.ids=bank.ids(key);self.mask=np.flatnonzero(np.isin(self.ids,support_ids))
  self.support_labels=np.asarray(support_labels).copy()
  if len(self.mask)!=len(self.support_labels):raise ValueError('Support ordering mismatch')
  self.events=[];self.queries=[];self.states={};self.costs=collections.Counter();self.physical=set();self.encoded=set();self.clusterings=set()
 def query(self,a,u,mode='train'):
  a=float(np.clip(a,0,1)) if self.bank.backend=='DBSCAN' else int(a);u=float(np.clip(u,0,1));start=time.perf_counter()
  params=self.bank.params(self.key,a,u);labels,meta=self.bank.predict(self.key,params)
  self.costs[f'{mode}_requests']+=1
  if meta['cache_hit']:self.costs['cache_hits']+=1
  else:self.costs['actual_fits']+=1;self.costs['fit_seconds']+=meta['seconds']
  self.physical.add(json.dumps(params,sort_keys=True));self.encoded.add((a,u));self.clusterings.add(labels.tobytes())
  e=dict(linkage_id=a,u=u,coords=[a,u],parameters=params,mode=mode,cache_hit=meta['cache_hit'],cache_source=meta['source'],
         seconds=time.perf_counter()-start,cumulative_queries=len(self.queries)+1,actual_fits=self.costs['actual_fits'])
  self.queries.append(e.copy())
  if mode=='train':
   e.update(score=float(nmi(self.support_labels,labels[self.mask])) if len(self.mask)>=2 else 0.,labels=labels);self.events.append(e)
  return labels
 def score(self):return self.events[-1]['score']
 def state(self,labels,a,u,bounds):
  key=labels.tobytes()
  if key not in self.states:
   center=self.bank.mean(self.ids);local=[]
   for label in np.unique(labels):
    ids=self.ids[labels==label];q=self.bank.mean(ids);distance=float(ball_distance(q[None],center[None])[0,0])/self.bank.meta['scale']
    local.append([distance,len(ids)/100]+list(self.bank.chart(q)))
   self.states[key]=local
  k=len(set(labels)-{-1})
  if self.bank.backend=='DBSCAN':global_state=[a,a-bounds[0][0],bounds[0][1]-a,u,u-bounds[1][0],bounds[1][1]-u,k/len(labels)]
  else:global_state=[u,u-bounds[0],bounds[1]-u,k/len(labels)]+[float(i==a) for i in range(4)]
  return [[global_state],self.states[key]]

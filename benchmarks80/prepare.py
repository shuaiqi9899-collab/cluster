"""Prepare only intrinsic hyperbolic geometry, partitions and native AC trees."""
import argparse,collections,hashlib,json,os,random,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from hyperbolic.geometry import ball_distance
from ac_joint.trees import LINKAGES,make_tree
from .common import OUT,ROOT,write,sha,manifest


def subset_key(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()[:24]


def representation(folder,row):
 raw=np.load(folder/'raw.npz');metadata=dict(used_labels=False,max_dimension=64,sampling=False)
 if (folder/'features_sparse.npz').exists():
  metadata['sparse_feature_sha256']=sha(folder/'features_sparse.npz')
  x=sparse.load_npz(folder/'features_sparse.npz').astype(float)
  d=min(64,x.shape[0]-1,x.shape[1]-1);svd=TruncatedSVD(d,random_state=20261105)
  x=StandardScaler().fit_transform(svd.fit_transform(x));metadata.update(transform='TruncatedSVD64 then standardize; sparse attributes',variance=float(svd.explained_variance_ratio_.sum()))
 else:
  x=np.asarray(raw['x'],dtype=float)
  keep=~np.isnan(x).all(axis=0);metadata['dropped_all_missing']=np.flatnonzero(~keep).tolist();x=x[:,keep]
  x=SimpleImputer(strategy='median').fit_transform(x)
  if row['B'] in ['2D shape and scale','FCPS structure','MARL shape benchmarks']:
   x=MinMaxScaler().fit_transform(x);metadata['scaling']='MinMax per coordinate, as shape reproduction'
  elif row.get('view_dimensions'):
   pieces=[];start=0
   for d in row['view_dimensions']:
    v=StandardScaler().fit_transform(x[:,start:start+d]);pieces.append(v/np.sqrt(d*len(row['view_dimensions'])));start+=d
   x=np.column_stack(pieces);metadata['scaling']='Per-view standardization and1/sqrt(d_view*V) equal-energy concatenation'
  else:x=StandardScaler().fit_transform(x);metadata['scaling']='Median imputation then standardization'
  nonconstant=np.ptp(x,axis=0)>0;metadata['dropped_constant']=np.flatnonzero(~nonconstant).tolist();x=x[:,nonconstant]
  if x.shape[1]>64:
   dim=min(64,len(x)-1);pca=PCA(dim,svd_solver='randomized',random_state=20261105);x=pca.fit_transform(x);metadata.update(transform='Unlabeled PCA64',variance=float(pca.explained_variance_ratio_.sum()))
 if not np.isfinite(x).all() or not x.shape[1]:raise ValueError('Degenerate/nonfinite feature representation')
 metadata.update(n=len(x),dimension=x.shape[1],raw_sha256=sha(folder/'raw.npz'))
 return x,raw['label_valid'],metadata


def entropy_partition(points,distance,seed):
 import hyperbolic.experiment as previous
 from ac_joint.prepare import auxiliary_groups
 from types import SimpleNamespace
 class ScalarConnectivity:
  def __init__(self,eps,min_samples):
   if eps!=.3 or min_samples!=1:raise ValueError('Unexpected auxiliary grouping')
  def fit_predict(self,x):return auxiliary_groups(np.asarray(x).reshape(-1))
 old=previous.DBSCAN
 try:
  previous.DBSCAN=ScalarConnectivity
  return previous.partition(SimpleNamespace(points=points,distances=distance),seed)
 finally:previous.DBSCAN=old


def prepare_dataset(out,key):
 out=Path(out);row=next(r for r in manifest(out) if r['key']==key)
 if row['status']!='ready':raise ValueError('Dataset is not admitted')
 folder=out/'prepared'/key;complete=folder/'preparation.json'
 if complete.exists() and json.loads(complete.read_text()).get('status')=='passed':return json.loads(complete.read_text())
 folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 inputfolder=out/'inputs'/key
 if not (folder/'geometry.json').exists():
  x,valid,meta=representation(inputfolder,row);mu=x.mean(0);z=x-mu;norm=np.linalg.norm(z,axis=1);radius=float(norm.max())
  if radius<=0:raise ValueError('Coincident feature vectors')
  a=np.arctanh(.5)/radius;factor=np.divide(np.tanh(a*norm),norm,out=np.full_like(norm,a),where=norm>0);points=z*factor[:,None]
  assert np.linalg.norm(points,axis=1).max()<=.5+1e-12
  np.save(folder/'points.npy',points);np.save(folder/'features.npy',x)
  d=np.lib.format.open_memmap(folder/'distances.npy',mode='w+',dtype=np.float64,shape=(len(x),len(x)))
  for block in range(0,len(x),256):d[block:block+256]=ball_distance(points[block:block+256],points)
  np.fill_diagonal(d,0);d.flush()
  rng=np.random.default_rng(20261105);pairs=rng.integers(0,len(x),size=(min(1000000,len(x)**2),2));lengths=d[pairs[:,0],pairs[:,1]];positive=lengths[lengths>0]
  scale=float(np.quantile(positive,.9));delta=float(np.min(d[d>0]))
  if x.shape[1]>=2:
   basis=PCA(2,random_state=20261105).fit(z).components_.T
  else:basis=np.column_stack([np.ones(1),np.zeros(1)])
  write(folder/'geometry.json',dict(space='H',curvature=-1,max_radius=.5,a=a,center=mu,scale=scale,delta=delta,
    eps_min=max(np.finfo(float).eps,delta/2),eps_max=scale,state_projection=basis,
    representation=meta,euclidean_clustering=False,euclidean_distance_matrix=False))
  del d
 points=np.load(folder/'points.npy');d=np.load(folder/'distances.npy',mmap_mode='r');n=len(points)
 valid=np.load(inputfolder/'raw.npz')['label_valid'];eligible=np.flatnonzero(valid)
 subsets={subset_key(np.arange(n)):np.arange(n)};regions={};partition_seconds=0.
 for seed in range(10):
  randomizer=random.Random(seed);order=np.array(randomizer.sample(range(n),n));supports=np.array(randomizer.sample(eligible.tolist(),int(.2*len(eligible))),dtype=int)
  seeds=folder/f'seed{seed}.npz'
  if not seeds.exists():np.savez_compressed(seeds,order=order,support_ids=np.sort(supports))
  prefile=folder/f'partition_seed{seed}.npz'
  if not prefile.exists():
   t=time.perf_counter();pre=entropy_partition(points,d,seed);pre['seconds']=time.perf_counter()-t;np.savez_compressed(prefile,**pre)
  pre=np.load(prefile);partition_seconds+=float(pre['seconds'])
  parts=[np.flatnonzero(pre['regions']==label) for label in np.unique(pre['regions'])];parts.sort(key=lambda ids:int(ids[0]))
  keys=[]
  for ids in parts:subsets[subset_key(ids)]=ids;keys.append(subset_key(ids))
  regions[str(seed)]=keys
 trees={};scales={name:0. for name in LINKAGES};build_seconds={name:0. for name in LINKAGES}
 for digest,ids in subsets.items():
  trees[digest]={};idsfile=folder/'subsets'/f'{digest}_ids.npy';idsfile.parent.mkdir(exist_ok=True)
  if not idsfile.exists():np.save(idsfile,ids)
  for linkage in LINKAGES:
   file=folder/'trees'/f'{digest}_{linkage}.npz';meta=file.with_suffix('.json');file.parent.mkdir(exist_ok=True)
   if not file.exists() or not meta.exists():
    local=np.asarray(d[np.ix_(ids,ids)]);z,h,m=make_tree(points[ids],local,linkage,hyperbolic=True)
    audit=m.pop('merge_audit',np.empty((0,4)));np.savez_compressed(file,z=z,heights=h,ids=ids,merge_audit=audit)
    write(meta,m);print('Prepared H tree',key,linkage,len(ids),f'{m["seconds"]:.2f}s',flush=True)
   m=json.loads(meta.read_text());scales[linkage]=max(scales[linkage],m['root_height']);build_seconds[linkage]+=m['seconds'];trees[digest][linkage]=str(file.relative_to(folder))
 result=dict(status='passed',space='H',key=key,n=n,global_key=subset_key(np.arange(n)),regions=regions,trees=trees,scales=scales,
  partition_seconds=partition_seconds,build_seconds=build_seconds,seconds=time.perf_counter()-start,unique_subsets=len(subsets),
  data_sha256=row['raw_sha256'],seeds=list(range(10)),representation=json.loads((folder/'geometry.json').read_text())['representation'])
 write(complete,result);return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();prepare_dataset(a.output,a.key)

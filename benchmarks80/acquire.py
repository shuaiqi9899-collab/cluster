"""Acquire data, validate versions, and document exclusions without sampling.

Remote artifacts are parsed as data; downloaded code is never executed.
"""
import argparse,gzip,hashlib,io,json,pickle,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import requests
from scipy.io import arff,loadmat
from scipy import sparse
from .common import OUT,ROOT,MAX_N,catalogue,write,sha

GAG='https://raw.githubusercontent.com/gagolews/clustering-data-v1/4199c2fb21e39bab604106f8589480e0ed758df6/'
DERIC='https://raw.githubusercontent.com/deric/clustering-benchmark/fc7aba73c873b0e017dd3a6d7113b1cb2ede118a/src/main/resources/datasets/real-world/'
JB='https://raw.githubusercontent.com/jbrownlee/Datasets/d20fcb6402ae34e653d4513b00f39257bb37ed7f/'
GAG_FILES={1:'other/iris',2:'uci/wine',4:'uci/glass',5:'uci/ecoli',6:'uci/yeast',7:'uci/wdbc',8:'uci/ionosphere',9:'uci/sonar',23:'sipu/flame',24:'sipu/jain',26:'sipu/spiral',27:'sipu/r15',29:'sipu/s1',30:'sipu/s2',31:'sipu/s3',32:'sipu/s4',33:'sipu/a1',34:'sipu/a2',35:'sipu/a3',36:'sipu/unbalance',39:'fcps/atom',40:'fcps/chainlink',41:'fcps/engytime',42:'fcps/hepta',43:'fcps/lsun',44:'fcps/target',45:'fcps/tetra',46:'fcps/twodiamonds',47:'fcps/wingnut'}
LOCAL={21:'Aggregation',22:'Compound',25:'Pathbased',28:'D31',49:'Asymmetric',50:'Skewed',51:'Cure-t1',52:'Cure-t2'}
ARFF={10:'haberman',11:'balance-scale',14:'segment'}
CSV={3:('wheat-seeds.csv',False),12:('breast-cancer-wisconsin.data',True),13:('banknote_authentication.csv',False)}
OFFICIAL_ZIP={16:(80,['optdigits.tra','optdigits.tes']),18:(146,['sat.trn','sat.tst']),19:(94,['spambase.data']),20:(107,['waveform.data']),60:(72,[])}


def fetch(url,folder,name=None):
 folder.mkdir(parents=True,exist_ok=True);p=folder/(name or url.split('/')[-1])
 if not p.exists():
  response=requests.get(url,timeout=(15,90),stream=True);response.raise_for_status()
  temp=p.with_suffix(p.suffix+'.download');size=0
  try:
   with temp.open('wb') as f:
    for block in response.iter_content(1024*1024):
     size+=len(block)
     if size>256*1024**2:raise MemoryError('Raw package exceeds256MiB per-artifact acquisition cap')
     f.write(block)
   temp.replace(p)
  finally:
   if temp.exists():temp.unlink()
 return p,dict(url=url,file=str(p.relative_to(folder.parent.parent)),sha256=sha(p),bytes=p.stat().st_size)


def encode_labels(y):
 y=np.asarray(y).reshape(-1)
 if y.dtype.kind in 'USO':
  y=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in y]);valid=~np.isin(y,['?','nan','None',''])
 else:valid=np.isfinite(y.astype(float))
 result=np.full(len(y),-1,dtype=np.int64);classes=np.unique(y[valid]);result[valid]=np.searchsorted(classes,y[valid])
 return result,valid,[str(c) for c in classes]


class DataUnpickler(pickle.Unpickler):
 """Whitelist only the ndarray/CSR constructors used by Planetoid data files."""
 def find_class(self,module,name):
  allowed={('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('scipy.sparse.csr','csr_matrix'),('scipy.sparse._csr','csr_matrix'),('_codecs','encode')}
  if (module,name) not in allowed:raise pickle.UnpicklingError(f'Forbidden data-pickle class {module}.{name}')
  return super().find_class(module,name)


def planetoid(which,folder):
 items={};sources=[]
 for part in ['x','tx','allx','y','ty','ally','graph','test.index']:
  url=f'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{which}.{part}'
  p,meta=fetch(url,folder);sources.append(meta)
  if part=='test.index':items[part]=np.loadtxt(p,dtype=int)
  else:
   with p.open('rb') as f:items[part]=DataUnpickler(f,encoding='latin1').load()
 order=items['test.index'];sorted_ids=np.sort(order);tx=items['tx'];ty=items['ty']
 if which=='citeseer':
  full=np.arange(sorted_ids.min(),sorted_ids.max()+1)
  extended=sparse.lil_matrix((len(full),tx.shape[1]));extended[sorted_ids-full[0]]=tx;tx=extended
  extended_y=np.zeros((len(full),ty.shape[1]));extended_y[sorted_ids-full[0]]=ty;ty=extended_y
 x=sparse.vstack((items['allx'],tx)).tolil();x[order]=x[sorted_ids]
 labels=np.vstack((items['ally'],ty));labels[order]=labels[sorted_ids]
 valid=labels.sum(axis=1)>0;y=labels.argmax(axis=1).astype(int);y[~valid]=-1
 # Preserve edges separately if obtained later; primary protocol uses attributes.
 return x.tocsr(),y,valid,sources,dict(node_features_only=True,edges_used=False,unlabeled_nodes=int((~valid).sum()))


def acquire_one(row,out):
 row=dict(row);number=row['id'];folder=out/'raw'/row['key'];sources=[];notes=[]
 if number in [56,57,58,59]:return dict(row,status='excluded_missing_exact_package',reason='Catalog explicitly lacks the exact PRIME preprocessed package; user authorized deferral')
 if row['n_declared']>MAX_N:return dict(row,status='excluded_resource',reason=f'Original N>{MAX_N}; no subsampling;4CPU/32GiB conservative full-data admission limit')
 canonical=out/'inputs'/row['key'];meta_file=canonical/'metadata.json'
 if meta_file.exists():
  prior=json.loads(meta_file.read_text())
  if prior.get('status')=='ready' and (canonical/'raw.npz').exists():
   if number in [76,77] and not any(s['url'].endswith('.graph') for s in prior['sources']):
    which='cora' if number==76 else 'citeseer';_,m=fetch(f'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{which}.graph',folder);prior['sources'].append(m);write(meta_file,prior)
   return prior
 try:
  if number in GAG_FILES:
   stem=GAG_FILES[number]
   p,m=fetch(GAG+stem+'.data.gz',folder);sources.append(m);x=np.loadtxt(gzip.open(p,'rt'),ndmin=2)
   p,m=fetch(GAG+stem+'.labels0.gz',folder);sources.append(m);raw_y=np.loadtxt(gzip.open(p,'rt'),dtype=int)
   y=raw_y-1;valid=np.ones(len(y),bool);label_names=[str(v) for v in np.unique(raw_y)]
   p,m=fetch(GAG+stem+'.txt',folder);sources.append(m)
   notes.append('Benchmark suite v1.2.0 labels0; original author reference partition, with0 mapped to noise-1')
  elif number in LOCAL:
   original=ROOT/'author_source/data'/f'Shape-{LOCAL[number]}.txt';folder.mkdir(parents=True,exist_ok=True);p=folder/original.name;shutil.copy2(original,p)
   raw=np.loadtxt(p);x=raw[:,:-1];y=raw[:,-1].astype(int);valid=np.ones(len(y),bool)
   sources=[dict(url='https://github.com/RingBDStack/AR-DBSCAN/tree/b9cfcd41fc1d34348e75352ec457dc1d08df8e02/data',file=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size,method='Reused original tracked author-data copy')]
   if number==52:
    assert len(y)==4200 and np.sum(y==-1)==200
    notes.append('Author file has4000 signal points+200 labeled noise points=4200; retained all points, no removal to match catalog4000')
  elif number in ARFF:
   p,m=fetch(DERIC+ARFF[number]+'.arff',folder);sources.append(m)
   data,description=arff.loadarff(p);names=data.dtype.names
   x=np.column_stack([data[k] for k in names[:-1]]).astype(float);y,valid,label_names=encode_labels(data[names[-1]])
  elif number in CSV:
   name,dropid=CSV[number];p,m=fetch(JB+name,folder);sources.append(m)
   raw=np.genfromtxt(p,delimiter=',',missing_values='?',filling_values=np.nan)
   x=raw[:,1:-1] if dropid else raw[:,:-1];y,valid,label_names=encode_labels(raw[:,-1])
  elif number==48:
   url='https://raw.githubusercontent.com/RingBDStack/AR-DBSCAN/b9cfcd41fc1d34348e75352ec457dc1d08df8e02/data/Shape-Unbalance2.txt'
   p,m=fetch(url,folder);sources.append(m);raw=np.loadtxt(p);x=raw[:,:-1];y=raw[:,-1].astype(int);valid=np.ones(len(y),bool)
  elif number==62:
   p,m=fetch('https://raw.githubusercontent.com/zmxu196/hubREP/main/dataset/3sources.mat',folder);sources.append(m);obj=loadmat(p)
   views=obj.get('X',obj.get('data'));target=next((obj[k] for k in ['Y','gt','label','y','truelabel'] if k in obj),None)
   if views is None or target is None:raise ValueError('Multi-view MAT schema not recognized: '+str([k for k in obj if not k.startswith('__')]))
   if np.asarray(target).dtype==object:
    targets=[np.asarray(t).reshape(-1) for t in target.reshape(-1)];assert all(np.array_equal(targets[0],t) for t in targets);target=targets[0]
   y,valid,label_names=encode_labels(target);arrays=[]
   for v in views.reshape(-1):
    v=v.toarray() if sparse.issparse(v) else np.asarray(v)
    if v.shape[0]!=len(y) and v.shape[1]==len(y):v=v.T
    if v.shape[0]!=len(y):raise ValueError('Unaligned view')
    arrays.append(v.astype(float))
   if len(arrays)!=3:raise ValueError('Not the3 complete views')
   x=np.column_stack(arrays);row['view_dimensions']=[a.shape[1] for a in arrays];notes.append('Three aligned views; per-view energy equalization before concatenation')
  elif number in [76,77]:
   x,y,valid,sources,extra=planetoid('cora' if number==76 else 'citeseer',folder);row.update(extra)
  elif number in OFFICIAL_ZIP:
   import zipfile
   uid,names=OFFICIAL_ZIP[number];p,m=fetch(f'https://archive.ics.uci.edu/static/public/{uid}/{row["K"].split("/")[-1]}.zip',folder,'official.zip');sources.append(m)
   with zipfile.ZipFile(p) as z:
    # Resolve expected official filenames, preserving splits; no zip extraction.
    if number==60:
     names=['mfeat-fou','mfeat-fac','mfeat-kar','mfeat-pix','mfeat-zer','mfeat-mor'];arrays=[]
     for name in names:
      f=next(k for k in z.namelist() if k.split('/')[-1]==name);arrays.append(np.loadtxt(io.BytesIO(z.read(f))))
     x=np.column_stack(arrays);y=np.repeat(np.arange(10),200);valid=np.ones(2000,bool);row['view_dimensions']=[a.shape[1] for a in arrays];notes.append('UCI documented class-block order,200 examples/class; all six aligned feature views')
    else:
     arrays=[]
     for name in names:
      f=next(k for k in z.namelist() if k.split('/')[-1]==name)
      content=z.read(f)
      if name.endswith('.Z'):raise ValueError('Compressed .Z requires explicit decoding')
      arrays.append(np.loadtxt(io.BytesIO(content),delimiter=',' if number in [16,19] else None))
     data=np.vstack(arrays);x=data[:,:-1];y,valid,label_names=encode_labels(data[:,-1])
  elif number in [66,67,68,69,70,79,80]:
   urls={66:'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2',
         67:'https://www.cs.columbia.edu/CAVE/databases/coil-20/coil-20-proc.zip',
         68:'https://www.cs.columbia.edu/CAVE/databases/coil-100/coil-100.zip',
         69:'https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip',
         70:'https://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.zip',
         79:'https://cs.uef.fi/sipu/datasets/g2-32-50.txt',80:'https://cs.uef.fi/sipu/datasets/dim064.txt'}
   p,m=fetch(urls[number],folder);sources.append(m)
   if number==66:
    import bz2
    from sklearn.datasets import load_svmlight_files
    q,m=fetch(urls[number].replace('usps.bz2','usps.t.bz2'),folder);sources.append(m)
    with bz2.open(p,'rb') as first,bz2.open(q,'rb') as second:xa,ya,xb,yb=load_svmlight_files([first,second],n_features=256)
    x=sparse.vstack([xa,xb]).tocsr();y,valid,label_names=encode_labels(np.r_[ya,yb])
   elif number in [79,80]:
    data=np.loadtxt(p);dim=32 if number==79 else 64
    if data.shape[1]!=dim+1:raise ValueError('Family file lacks explicit reference labels or expected dimensions')
    x=data[:,:-1];y,valid,label_names=encode_labels(data[:,-1]);notes.append('Fixed family configuration: '+('G2 d32 parameter50' if number==79 else 'DIM d64'))
   else:
    import zipfile,re
    from PIL import Image
    with zipfile.ZipFile(p) as z:
     entries=[name for name in z.namelist() if name.lower().endswith(('.png','.pgm','.gif'))]
     arrays=[];classes=[]
     for name in sorted(entries):
      image=Image.open(io.BytesIO(z.read(name))).convert('RGB' if number==68 else 'L');arrays.append(np.asarray(image).reshape(-1))
      match=re.search(r'obj(\d+)',name) if number in [67,68] else re.search(r'/s(\d+)/', '/'+name) if number==69 else re.search(r'subject(\d+)',name)
      if match is None:raise ValueError('Image identity not encoded by documented filename')
      classes.append(int(match.group(1)))
     x=np.asarray(arrays);y,valid,label_names=encode_labels(classes)
  else:
   # Public entries may exist, but no validated matching artifact has been
   # located yet. Do not substitute arbitrary processed versions.
   return dict(row,status='excluded_exact_package_not_located',reason='No verified matching artifact found in catalog repositories/accessible mirrors; no version substitution')
  if len(y)!=row['n_declared'] and number!=52:raise ValueError(f'N mismatch: catalog{row["n_declared"]}, observed{len(y)}')
  if row.get('E','').isdigit() and x.shape[1]!=int(row['E']):raise ValueError(f'd mismatch: catalog{row["E"]}, observed{x.shape[1]}')
  if len(y)>MAX_N:raise MemoryError('Actual sample count exceeds resource admission limit')
  if x.shape[0]!=len(y):raise ValueError('Feature-label alignment mismatch')
  if np.any(np.isinf(x.data if sparse.issparse(x) else x)):raise ValueError('Infinite feature values')
  if np.sum(valid)<10:raise ValueError('Insufficient labeled examples for ten-seed protocol')
  # Reference counts are used only to identify the downloaded version, never
  # passed into a backend, parameter domain, reward target or stopping rule.
  if str(row.get('G','')).isdigit() and len(np.unique(y[valid]))!=int(row['G']):
   raise ValueError(f'Reference-label count mismatch: expected{row["G"]}, observed{len(np.unique(y[valid]))}')
  canonical.mkdir(parents=True,exist_ok=True)
  if sparse.issparse(x):
   sparse.save_npz(canonical/'features_sparse.npz',x);np.savez_compressed(canonical/'raw.npz',y=y,label_valid=valid,ids=np.arange(len(y)))
  else:np.savez_compressed(canonical/'raw.npz',x=x,y=y,label_valid=valid,ids=np.arange(len(y)))
  result=dict(row,status='ready',n_actual=len(y),d_actual=x.shape[1],sources=sources,notes=notes,
              label_count=int(len(np.unique(y[valid]))),labeled_count=int(np.sum(valid)),raw_sha256=sha(canonical/'raw.npz'))
  write(meta_file,result);return result
 except requests.exceptions.HTTPError as exc:
  code=exc.response.status_code if exc.response is not None else None
  return dict(row,status='blocked_network' if code==403 else 'excluded_exact_package_not_located',reason=str(exc),sources=sources)
 except requests.exceptions.RequestException as exc:return dict(row,status='blocked_network',reason=str(exc),sources=sources)
 except MemoryError as exc:return dict(row,status='excluded_resource',reason=str(exc),sources=sources)
 except Exception as exc:return dict(row,status='blocked_version_or_format',reason=repr(exc),sources=sources)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=catalogue();results=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(acquire_one,row,a.output) for row in rows]
  for future in as_completed(futures):
   r=future.result();results.append(r);results.sort(key=lambda r:r['id']);write(a.output/'datasets.json',results)
   print(r['id'],r['name'],r['status'],r.get('reason',''),flush=True)
 write(a.output/'acquisition.json',dict(catalog_rows=80,max_n=MAX_N,sampling=False,space='H',datasets=results))


if __name__=='__main__':main()

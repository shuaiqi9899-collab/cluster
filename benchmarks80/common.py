import hashlib,json,re
from pathlib import Path
from hyperbolic.prepare import ROOT

OUT=ROOT/'results/benchmarks80'
MAX_N=10000
BACKENDS=('DBSCAN','AC','HDBSCAN')
METHODS=['MA-TD3','SA-TD3','Rand','BO-TPE','Anneal','PSO','GA','DE','MA-FrozenActor','SA-FrozenActor','MA-Random','Grid-30']


def methods(backend):return METHODS[:-1] if backend=='DBSCAN' else METHODS

def write(path,value):
 import os
 import numpy as np
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 def cast(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=cast,allow_nan=False));temp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def catalogue():
 rows=json.loads((ROOT/'benchmarks80/catalog.json').read_text())
 for row in rows:
  row['id']=int(row['A']);row['name']=row['C'];row['n_declared']=int(row['D'])
  row['key']=f'{row["id"]:02d}-'+re.sub('[^a-z0-9]+','-',row['C'].lower()).strip('-')
 return rows

def manifest(out=OUT):return json.loads((out/'datasets.json').read_text())


def training_sources():
 files=[ROOT/'benchmarks80'/name for name in ['common.py','context.py','prepare.py','search_db.py','experiment.py']]
 files += [ROOT/p for p in ['ac_joint/search.py','ac_joint/network.py','ac_joint/trees.py','ac_joint/prepare.py','hyperbolic/geometry.py','hyperbolic/experiment.py','density_cf/hdb.py']]
 files += list((ROOT/'author_source').rglob('*.py'))
 return {str(p.relative_to(ROOT)):sha(p) for p in files}

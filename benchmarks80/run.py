"""Durable H-only job queue with source gates and resource deferrals."""
import argparse,fcntl,hashlib,json,os,signal,shutil,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from .common import OUT,ROOT,MAX_N,BACKENDS,methods,write,manifest,training_sources,sha
from .status import status


def run(out,workers=3,prep_timeout=1800,case_timeout=900,rss_gib=8,datasets=None):
 out=Path(out);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=manifest(out);ready=sorted([r for r in rows if r['status']=='ready'],key=lambda r:(r['n_actual'],r['id']))
 if datasets is not None:
  unknown=set(datasets)-{r['key'] for r in ready}
  if unknown:raise ValueError('Unknown or unready datasets: '+str(unknown))
  ready=[r for r in ready if r['key'] in datasets]
 protocol_file=out/'protocol.json';sources=training_sources();data_hashes={r['key']:r['raw_sha256'] for r in ready}
 if protocol_file.exists():
  protocol=json.loads(protocol_file.read_text())
  if protocol['training_sources']!=sources:raise RuntimeError('Frozen training sources changed; preserve results and use a new output/protocol version')
  for key,digest in protocol['data_hashes'].items():
   if key in data_hashes and digest!=data_hashes[key]:raise RuntimeError('Frozen dataset changed: '+key)
  protocol['data_hashes'].update(data_hashes)
 else:
  protocol=dict(version='H80-v1',space='H',curvature=-1,max_radius=.5,euclidean_clustering=False,euclidean_distance_matrix=False,
   max_n=MAX_N,sampling=False,seeds=list(range(10)),backends=BACKENDS,methods={b:methods(b) for b in BACKENDS},
   batch=16,layers=3,max_episodes=14,max_steps=30,display='First30 training requests incl4 probes/cache hits; complete training and all detect costs retained',
   dbscan='Two numeric coordinates: log native-H eps from delta/2 to H-Q90; linear rounded min_samples1..ceil(n_local/4)',
   ac='Nominal single/complete/average/native Frechet-SSE Ward plus linear normalized native-height threshold',
   hdbscan='Nominal min_samples1/5/10/20 capped n-1; log min_cluster_size2..ceil(n_local/4); native precomputed H, EOM, alpha1, epsilon0, root disallowed',
   support='20% of labeled sample IDs, random.Random(seed) after full order draw; canonical point-ID fitting; no class balancing; labels only in reward/evaluation',
   representation='Shapes retain dimensions; other inputs unlabeled impute/standardize/PCA or sparse SVD to max64; policy position projection2D only',
   region_policy='Author SE/k scan with native H kNN; scalar uncertainty connected groups, not Euclidean clustering; all ten partitions and four AC families prepared before training',
   resource_policy='Data prep or full cases exceeding budgets are deferred, not truncated or silently approximated; all partial artifacts retained',
   data_hashes=data_hashes,training_sources=sources)
 protocol['training_signature']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
 write(protocol_file,protocol)
 event=threading.Event();processes={};process_lock=threading.Lock();finished=0
 def terminate(process):
  if process.poll() is None:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
 def defer(key,reason):
  p=out/'execution'/(key+'.json');old=json.loads(p.read_text()) if p.exists() else {}
  old.update(dataset=key,status='deferred_resource',reason=reason,updated_unix=time.time());write(p,old)
 def task(row):
  key=row['key'];statefile=out/'execution'/(key+'.json')
  if statefile.exists() and json.loads(statefile.read_text()).get('status')=='completed':return key,'already_completed'
  if event.is_set():return key,'interrupted'
  # Preserve interrupted attempts before a resource-profile retry.
  for folder in (out/'cases').glob('*/'+key+'/seed*/*'):
   if folder.is_dir() and not (folder/'result.json').exists() and any(folder.iterdir()):
    archive=out/'interrupted_attempts'/folder.relative_to(out/'cases');archive=archive.with_name(archive.name+f'.{time.time_ns()}');archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(archive))
  logpath=out/'logs'/(key+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
  with logpath.open('a') as log:
   process=subprocess.Popen([sys.executable,'-u','-m','benchmarks80.worker',key,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   with process_lock:processes[key]=process
   while process.poll() is None:
    if event.is_set():terminate(process);return key,'interrupted'
    try:state=json.loads(statefile.read_text()) if statefile.exists() else {}
    except json.JSONDecodeError:state={}
    elapsed=time.time()-state.get('phase_started_unix',time.time())
    if state.get('status')=='preparing' and elapsed>prep_timeout:
     terminate(process);defer(key,f'Full native preparation exceeded{prep_timeout}s; no sampled/tangent/Euclidean fallback');return key,'deferred_resource'
    if state.get('status')=='training' and time.time()-state.get('case_started_unix',time.time())>case_timeout:
     terminate(process);defer(key,f'One complete-training case exceeded{case_timeout}s; dataset deferred, partial results are not a completed comparison');return key,'deferred_resource'
    try:
     line=next(x for x in Path(f'/proc/{process.pid}/status').read_text().splitlines() if x.startswith('VmRSS:'));rss=int(line.split()[1])*1024
     if rss>rss_gib*1024**3:
      terminate(process);defer(key,f'Worker RSS exceeded{rss_gib}GiB budget');return key,'deferred_resource'
    except (OSError,StopIteration):pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   with process_lock:processes.pop(key,None)
   if process.returncode not in [0,75]:
    old=json.loads(statefile.read_text()) if statefile.exists() else {}
    if old.get('status')!='failed':old.update(dataset=key,status='failed',reason=f'Worker exited{process.returncode}',updated_unix=time.time());write(statefile,old)
   return key,'completed' if process.returncode==0 else 'deferred_resource' if process.returncode==75 else 'failed'
 def interrupted(signum,frame):event.set();raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  with ThreadPoolExecutor(max_workers=workers) as pool:
   futures={pool.submit(task,row):row['key'] for row in ready};last_report=0.
   while futures:
    done,_=wait(futures,timeout=20,return_when=FIRST_COMPLETED)
    for future in done:
     key,phase=future.result();futures.pop(future);finished+=1;print(key,phase,flush=True)
    if time.time()-last_report>=20 or done:
     current=status(out);current.update(phase='running',supervisor_pid=os.getpid(),updated_unix=time.time(),workers=workers,prep_timeout=prep_timeout,case_timeout=case_timeout,rss_gib=rss_gib)
     write(out/'run_status.json',current);print(json.dumps(current),flush=True);last_report=time.time()
    # Refresh lightweight summaries without refitting any model.
    if done:
     from .analyze import analyze
     analyze(out,plots=False)
  from .analyze import analyze
  analyze(out,plots=True);current=status(out);current.update(phase='completed_with_failures' if current['failed_datasets'] else 'completed_with_deferrals' if current['deferred_resource_datasets'] else 'completed',supervisor_pid=os.getpid(),actually_running=False,running_pids=[],updated_unix=time.time());write(out/'run_status.json',current)
 except BaseException:
  event.set()
  with process_lock:
   for process in processes.values():terminate(process)
  current=status(out);current.update(phase='interrupted_or_failed',updated_unix=time.time());write(out/'run_status.json',current);raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=3);p.add_argument('--prep-timeout',type=int,default=1800);p.add_argument('--case-timeout',type=int,default=900);p.add_argument('--rss-gib',type=float,default=8);p.add_argument('--datasets',nargs='+');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if a.dry_run:print(json.dumps(status(a.output)))
 else:run(a.output,a.workers,a.prep_timeout,a.case_timeout,a.rss_gib,a.datasets)

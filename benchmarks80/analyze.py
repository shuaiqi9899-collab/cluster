"""H-only Fig6 pages, held-out curves, acquisition and completion accounting."""
import argparse,collections,csv,fcntl,json,os
from pathlib import Path
import numpy as np
from .common import OUT,BACKENDS,methods,write,manifest


def analyze(out=OUT,plots=True):
 out=Path(out);analysis_lock=(out/'analysis.lock').open('a');fcntl.flock(analysis_lock,fcntl.LOCK_EX)
 rows=manifest(out);dataset_index={r['key']:r for r in rows};ready=[r for r in rows if r['status']=='ready']
 dest=out/'analysis';dest.mkdir(parents=True,exist_ok=True)
 records=[];failures=[]
 for p in (out/'cases').glob('*/*/seed*/*/result.json'):
  try:r=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  if r['status']=='ok':records.append(r)
  else:failures.append(str(p))
 work={}
 for p in (out/'execution').glob('*.json'):
  try:work[p.stem]=json.loads(p.read_text())
  except json.JSONDecodeError:continue
 available=collections.defaultdict(set)
 for r in records:available[r['dataset']].add((r['backend'],r['seed'],r['method']))
 required={(b,s,m) for b in BACKENDS for s in range(10) for m in methods(b)}
 fully_complete={key for key,state in work.items() if state.get('status')=='completed' and required.issubset(available[key])}
 summary=[]
 for dataset in ready:
  for backend in BACKENDS:
   for method in methods(backend):
    group=[r for r in records if (r['dataset'],r['backend'],r['method'])==(dataset['key'],backend,method)]
    if not group:continue
    item=dict(id=dataset['id'],dataset=dataset['name'],key=dataset['key'],backend=backend,method=method,seeds=len(group),n=group[0]['n'],space='H',execution_status=work.get(dataset['key'],{}).get('status','pending'))
    for m in ['heldout_nmi30','heldout_ari30','full_training_heldout_nmi','noise_fraction','clusters','agents','all_training_requests','total_queries','actual_fits','actor_updates','critic_updates','seconds','cpu_seconds']:
     values=[r[m] for r in group];item[m+'_mean']=float(np.mean(values));item[m+'_std']=float(np.std(values,ddof=1)) if len(group)>1 else 0.
    summary.append(item)
 if summary:
  with (dest/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 table=['# 80条目录：双曲空间 DBSCAN / AC / HDBSCAN','',
 '仅运行H空间；曲率−1，最大半径0.5。超过10000点暂缓，无抽样；精确包未确定与网络受阻分别记录。BIRCH算法保持淘汰。',
 '每个纳入数据集正式目标为10个种子；DBSCAN11、AC12、HDBSCAN12种搜索方法，共350例。预跑不计入正式完成数。完整TD3训练，回顾性展示前30次请求，支持奖励选参，补集只作离线评价。','',
 '## 数据下载与版本核对','', '| ID | 数据集 | 原表N / 实际N | 数据状态 | 说明 |','|---:|---|---:|---|---|']
 for r in rows:table.append(f'| {r["id"]} | {r["name"]} | {r["n_declared"]} / {r.get("n_actual","—")} | {r["status"]} | '+r.get('reason','; '.join(r.get('notes',[]))).replace('|','/').replace('\n',' ')+' |')
 table+=['','## 正式运行状态','',f'输入已就绪{len(ready)}项，候选目标{len(ready)*350}例；已保存成功{len(records)}例、失败{len(failures)}例。资源检查后暂缓的任务另见下表，不计为完成。',
 '', '| 数据集 | 执行阶段 | 已保存案例 | 目标 |','|---|---|---:|---:|']
 for r in ready:
  n=sum(x['dataset']==r['key'] for x in records);state=work.get(r['key'],{})
  table.append(f'| {r["name"]} | {state.get("status","pending")} | {n} | 350 |')
 comparisons=[]
 for dataset in ready:
  if dataset['key'] not in fully_complete:continue
  group={(r['backend'],r['seed']):r for r in records if r['dataset']==dataset['key'] and r['method']=='MA-TD3'}
  if not all((b,s) in group for b in BACKENDS for s in range(10)):continue
  item=dict(id=dataset['id'],dataset=dataset['name'],n=dataset['n_actual'])
  for backend in BACKENDS:
   values=[group[backend,s]['heldout_nmi30'] for s in range(10)];item[backend+'_mean']=float(np.mean(values));item[backend+'_std']=float(np.std(values,ddof=1))
  for a,b in [('AC','DBSCAN'),('HDBSCAN','DBSCAN'),('HDBSCAN','AC')]:item[a+'_minus_'+b]=float(np.mean([group[a,s]['heldout_nmi30']-group[b,s]['heldout_nmi30'] for s in range(10)]))
  comparisons.append(item)
 if comparisons:
  with (dest/'backend_comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 table+=['','## 指标与成本','',
 '普通NMI的参考标签未必代表密度簇；噪声、单例和最大簇比例应共同解释。CiteSeer无参考标签节点保留参与聚类但不进入支持奖励及外部评价，支持比例按有标签点计算。',
 '同名近缘版本不是独立来源；Unbalance与Unbalance2分别列出但不作为独立证据重复推断。主比较只在每组十种子齐备后进行。',
 '图中的全标签历史最大NMI只用于Fig.6式展示，补集曲线为支持奖励incumbent、可不单调；完整训练结果与前30请求结果分开记录。',
 '高维数据的固定无标签PCA/SVD表示保留至多64维，网络位置状态的二维投影不参与聚类距离。全部邻域、SE图和AC/HDBSCAN层次采用双曲距离；未生成欧氏距离矩阵或欧氏聚类。','',
 '[汇总CSV](summary.csv)；[三后端十种子配对均值比较](backend_comparisons.csv)（仅纳入已完成且已核验的数据集）。','']
 counts=collections.Counter(r['status'] for r in rows)
 write(dest/'status.json',dict(catalog=80,data_statuses=dict(counts),ready_datasets=len(ready),candidate_cases=len(ready)*350,successful_cases=len(records),failed_cases=failures,space='H',seeds=list(range(10)),sampling=False))
 if plots and records:
  os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'));os.environ.setdefault('XDG_CACHE_HOME',str(out/'plot-cache'))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  present=[r for r in ready if r['key'] in fully_complete]
  for backend in BACKENDS:
   displayed=methods(backend)[:8];colors=dict(zip(displayed,plt.cm.tab10(np.linspace(0,.9,len(displayed)))))
   for page,start in enumerate(range(0,len(present),8),1):
    chunk=present[start:start+8]
    for metric,prefix,label in [('curve','fig6','Full-label historical maximum NMI'),('heldout_curve','heldout','Held-out NMI, support-selected incumbent')]:
     fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
     for ax,dataset in zip(axes.flat,chunk):
      for method in displayed:
       group=[r for r in records if (r['dataset'],r['backend'],r['method'])==(dataset['key'],backend,method)]
       if not group:continue
       values=np.array([r[metric] for r in group]);mean=values.mean(0);sd=values.std(0)
       ax.plot(range(1,31),mean,label=method,color=colors[method],linestyle='--' if method=='SA-TD3' else '-')
       if method=='MA-TD3':ax.fill_between(range(1,31),np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.12,color=colors[method])
      count=sum((r['dataset'],r['backend'],r['method'])==(dataset['key'],backend,'MA-TD3') for r in records)
      ax.set_title(f'{dataset["name"]} (MA seeds={count}/10)',fontsize=10);ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.set_xlabel('Training requests / region')
     for ax in list(axes.flat)[len(chunk):]:ax.axis('off')
     for ax in axes[:,0]:ax.set_ylabel(label)
     handles,names=axes.flat[0].get_legend_handles_labels()
     if handles:fig.legend(handles,names,ncol=8,loc='upper center')
     fig.suptitle(f'{backend}, H only, page{page}; retrospective first30 requests',y=.94);fig.tight_layout(rect=[0,0,1,.92])
     name=f'{prefix}_{backend}_H_page{page:02d}';fig.savefig(dest/(name+'.png'),dpi=150);fig.savefig(dest/(name+'.pdf'));plt.close(fig)
     if prefix=='fig6':table.append(f'- [{backend} 双曲主图 第{page}页]({name}.png)')
 (dest/'报告.md').write_text('\n'.join(table));print('Analysis',len(records),'successful cases',flush=True)
 analysis_lock.close()
 return dict(successful=len(records),failures=failures)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--no-plots',action='store_true');a=p.parse_args();analyze(a.output,not a.no_plots)

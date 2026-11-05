"""Offline summaries; incomplete matrices cannot establish similarity."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from hyperbolic.prepare import ROOT,DATASETS,write_json
from .experiment import METHODS,SPACES,methods_for
from .lifecycle import ACTIVE_BACKENDS,RETIRED_BACKENDS


def analyze(out,publish=True):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=['# HDBSCAN：欧氏—双曲十种子实验','','## 当前实验范围','',
            '按用户指示，BIRCH已终止并淘汰，不再续跑，也不纳入正式比较。其538个已完成案例和原报告仅保留归档；剩余2262例已取消。当前新增后端仅保留HDBSCAN，既有DBSCAN与AC结果保留。',
            'HDBSCAN目标2800例，覆盖八个数据集、seed0–9、E/H主比较、H固定E区域及弱/强映射对照。沿用20%支持标签、冻结样本ID和SE区域，补集标签只在离线评价中使用。',
            '主双曲映射使用曲率−1的Poincaré球、最大点半径0.5；弱/强映射为0.2/0.8。沿用既有映射和几何缓存，不把所有点压到等半径球面。',
            '搜索min_samples∈{1,5,10,20}（局部截断到n−1）和min_cluster_size；u∈[0,1]对数映射到[2,max(2,ceil(n_local/4))]并取整。不搜索真实簇数K。',
            '固定hdbscan0.8.40、generic/precomputed原生距离、EOM、alpha1、epsilon0、allow_single_cluster=False；min_samples口径不含自身，噪声保持−1，合并区域时仅偏移非噪声簇ID。',
            'EOM稳定性依赖逆距离，非线性单调距离变换仍可能改变选簇，不能仅凭距离排序不变断言输出相同。',
            '三层独立TD3网络，最多14局/层、每局30步、batch16。沿用完整训练，回顾性展示每区域前30次训练请求，含4类探针及缓存命中；detect、后续训练及网络更新成本另存。',
            '基线每区域30请求。PSO/GA/DE枚举四类min_samples后在各类内优化u，pop4、配额8/8/7/7按seed轮换。支持NMI选参，补集不参与选择。',
            '主比较预设标准：MA-TD3的H−E补集NMI宏平均90%配对bootstrap区间完全落在±0.02内，且至少7/8数据集均值差落在±0.05内。未完成完整配对时不作相近判定。','']
    statuses={}
    for backend in ACTIVE_BACKENDS:
        dest=out/backend/'analysis';dest.mkdir(parents=True,exist_ok=True)
        rows=[json.loads(p.read_text()) for p in sorted((out/backend/'cases').glob('*/*/result.json'))]
        valid=[r for r in rows if r['status']=='ok'];failed=[r for r in rows if r['status']!='ok']
        for r in valid:
            r['unique_physical_parameters']=sum(t['unique_physical_parameters'] for t in r['regions'])
            for metric in ['fit_seconds','hierarchy_new_seconds']:
                r[metric]=sum(t['costs'].get(metric,0.) for t in r['regions'])
        index={(r['space'],r['dataset'],r['method'],r['seed']):r for r in valid}
        status=dict(expected=2800,passed=len(valid),failed=len(failed),counts={s:sum(r['space']==s for r in valid) for s in SPACES})
        statuses[backend]=status;write_json(dest/'status.json',status)
        summary=[]
        metrics=['heldout_nmi30','heldout_ari30','full_training_heldout_nmi','noise_fraction','clusters',
                 'singleton_fraction','largest_cluster_fraction','agents','all_training_requests',
                 'total_queries','actual_fits','unique_parameter_pairs','unique_clusterings','actor_updates','critic_updates','seconds','cpu_seconds']
        metrics+=['unique_physical_parameters','fit_seconds','hierarchy_new_seconds']
        for space in SPACES:
            for dataset in DATASETS:
                for method in methods_for(space):
                    group=[r for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                    if not group:continue
                    item=dict(backend=backend,space=space,dataset=dataset,method=method,runs=len(group))
                    for metric in metrics:
                        values=[r[metric] for r in group];item[metric+'_mean']=float(np.mean(values));item[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                    summary.append(item)
        if summary:
            with (dest/'summary.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        pairs=[]
        for dataset in DATASETS:
            for method in METHODS:
                seeds=[s for s in range(10) if ('E',dataset,method,s) in index and ('H',dataset,method,s) in index]
                if seeds:
                    differences=[index['H',dataset,method,s]['heldout_nmi30']-index['E',dataset,method,s]['heldout_nmi30'] for s in seeds]
                    pairs.append(dict(dataset=dataset,method=method,seeds=seeds,differences=differences,mean_difference=float(np.mean(differences))))
        write_json(dest/'paired_comparisons.json',pairs)
        primary=[r for r in pairs if r['method']=='MA-TD3']
        complete=len(primary)==8 and all(len(r['seeds'])==10 for r in primary)
        equivalence=dict(complete=complete,status='incomplete',margin=.02)
        if complete:
            diffs=np.array([r['differences'] for r in primary]);rng=np.random.default_rng(20261105)
            sampled=rng.integers(0,10,size=(10000,8,10));boot=np.take_along_axis(diffs[None],sampled,axis=2).mean((1,2))
            lo,hi=np.quantile(boot,[.05,.95]);within=sum(abs(r['mean_difference'])<=.05 for r in primary)
            equivalence.update(mean_difference=float(diffs.mean()),ci90=[lo,hi],datasets_within_005=within,
                               status='met_predefined_similarity' if lo>-.02 and hi<.02 and within>=7 else 'did_not_meet_predefined_similarity')
        write_json(dest/'equivalence.json',equivalence)
        report += [f'## {backend}','',f'正式完整矩阵目标2800例；当前成功{len(valid)}例，失败{len(failed)}例。各空间完成数：{status["counts"]}。','']
        if complete:
            lo,hi=equivalence['ci90'];conclusion='达到' if equivalence['status']=='met_predefined_similarity' else '未达到'
            report.append(f'MA-TD3：H−E补集NMI宏平均差{equivalence["mean_difference"]:.4f}，90%区间[{lo:.4f},{hi:.4f}]，{equivalence["datasets_within_005"]}/8数据集差值在±0.05内；**{conclusion}预设相近标准**。')
        else:report.append('完整十种子配对尚未完成，当前值仅为已完成运行的结果，不能据此判定双曲与欧氏效果相近。')
        report+=['','| 数据集 | E补集NMI | H补集NMI | H固定E区域 | H弱 | H强 |','|---|---:|---:|---:|---:|---:|']
        for dataset in DATASETS:
            cells=[]
            for space in SPACES:
                group=[r['heldout_nmi30'] for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
                cells.append(f'{np.mean(group):.4f} (n={len(group)})' if group else '未完成')
            report.append('| '+dataset+' | '+' | '.join(cells)+' |')
        report+=['','| 空间 | 方法 | 例数 | 前30补集NMI | 全训练补集NMI | 全查询均值 | Actor更新均值 | 噪声率 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for space in ['E','H']:
            for method in METHODS:
                group=[r for r in valid if (r['space'],r['method'])==(space,method)]
                if group:
                    values=[np.mean([r[m] for r in group]) for m in ['heldout_nmi30','full_training_heldout_nmi','total_queries','actor_updates','noise_fraction']]
                    report.append(f'| {space} | {method} | {len(group)} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
        colors=dict(zip(METHODS,plt.cm.tab20(np.linspace(0,1,len(METHODS)))))
        displayed=METHODS[:8]+['Grid-30']
        for space in ['E','H']:
            for metric,label,prefix in [('curve','Full-label historical maximum NMI','fig6'),('heldout_curve','Held-out NMI of support-selected incumbent','heldout')]:
                fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
                for ax,dataset in zip(axes.flat,DATASETS):
                    for method in displayed:
                        group=[r for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                        if not group:continue
                        values=np.array([r[metric] for r in group]);mean=values.mean(0);sd=values.std(0)
                        ax.plot(range(1,31),mean,label=method,color=colors[method],linestyle='--' if method in ['SA-TD3','Grid-30'] else '-')
                        if method=='MA-TD3':ax.fill_between(range(1,31),np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.12,color=colors[method])
                    count=sum((space,dataset,'MA-TD3',s) in index for s in range(10))
                    ax.set_title(f'{dataset} (MA n={count})');ax.set_ylim(0,1.02);ax.grid(alpha=.25);ax.set_xlabel('Training candidate requests / region')
                for ax in axes[:,0]:ax.set_ylabel(label)
                handles,labels=axes.flat[0].get_legend_handles_labels()
                if handles:fig.legend(handles,labels,ncol=9,loc='upper center')
                fig.suptitle(f'{backend} {space}; first30 retrospective requests, complete training costs separate',y=.94)
                fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(dest/f'{prefix}_{backend}_{space}.png',dpi=150)
                fig.savefig(dest/f'{prefix}_{backend}_{space}.pdf');plt.close(fig)
        report+=['',f'[欧氏Fig.6]({backend}/fig6_{backend}_E.png) · [双曲Fig.6]({backend}/fig6_{backend}_H.png)',
                 f'[欧氏补集曲线]({backend}/heldout_{backend}_E.png) · [双曲补集曲线]({backend}/heldout_{backend}_H.png)',
                 f'[完整汇总CSV]({backend}/summary.csv) · [配对差值]({backend}/paired_comparisons.json)','']
    report+=['## 解释与成本限制','',
             '区域独立聚类、不会跨区域合并；HDBSCAN可把整个区域判为噪声。需要同时阅读NMI、ARI、噪声率、单例率与最大簇占比。',
             '支持点不足或仅含一个支持类别时，局部监督信号可能退化；各区域记录support_count与support_class_count，不把局部奖励1解释为全局聚类正确。',
             '完整训练只依据支持奖励继续选择参数，因此可能提高支持奖励却降低补集NMI；没有使用补集标签回退或筛选完整训练结果。',
             '全标签历史最大NMI仅为Fig.6式离线展示，不能用于选参数；主指标为支持奖励选出的第30请求incumbent在补集上的NMI。补集曲线不强制单调。',
             '完整训练多于30请求，不能把全训练最优结果与30请求基线当作同等预算比较。所有请求、实际拟合数、不同物理参数、不同分区和网络更新数均另存。缓存跨方法共享，墙钟不代表独立冷启动算法效率。',
             '这些实验仍是二维欧氏形状数据的受控双曲迁移，不能证明算法在天然双曲数据上的优势。',
             '新增代码与输出分别位于density_cf/和results/density_cf/，旧DBSCAN与AC的源码、结果和报告不改写。','']
    validation=out/'validation.json'
    if validation.exists():report+=['## 核验','',f'核验记录：[validation.json](validation.json)。状态：{json.loads(validation.read_text())["status"]}，已核验{json.loads(validation.read_text())["verified_cases"]}例；核验范围以该记录为准。','']
    audit=out/'backend_audit.json'
    if audit.exists():
        report += [f'选定后端模型另行核对：[backend_audit.json](backend_audit.json)。{json.loads(audit.read_text())}。','']
    retirement=out/'retirement.json'
    if retirement.exists():report+=['BIRCH终止与归档记录：[retirement.json](retirement.json)。','']
    readiness=out/'readiness.json'
    if readiness.exists():report += ['当前HDBSCAN测试、运行范围与历史归档记录见[readiness.json](readiness.json)。','']
    (out/'报告.md').write_text('\n'.join(report));write_json(out/'status.json',statuses)
    if publish:
        dest=ROOT/'density_cf/report';dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/'报告.md',dest/'报告.md');shutil.copy2(out/'status.json',dest/'status.json')
        if validation.exists():shutil.copy2(validation,dest/'validation.json')
        if audit.exists():shutil.copy2(audit,dest/'backend_audit.json')
        if readiness.exists():shutil.copy2(readiness,dest/'readiness.json')
        if retirement.exists():shutil.copy2(retirement,dest/'retirement.json')
        for backend in statuses:
            shutil.copytree(out/backend/'analysis',dest/backend,dirs_exist_ok=True)
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'density_cf').glob('*') if p.is_file()}
        artifacts={str(p.relative_to(dest)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                   for p in dest.rglob('*') if p.is_file() and p.name!='source_manifest.json'}
        write_json(dest/'source_manifest.json',dict(source_sha256=files,report_files=artifacts,results_root=str(out),status=statuses))
    print(json.dumps(statuses),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/density_cf');p.add_argument('--no-publish',action='store_true')
    a=p.parse_args();analyze(a.output.resolve(),not a.no_publish)

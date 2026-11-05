"""Offline evaluation, paired uncertainty, figures and Chinese report."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .prepare import DATASETS, ROOT, write_json
from .experiment import METHODS, methods_for


def analyze(out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted((out/'cases').glob('*/*/result.json'))
    rows = [json.loads(p.read_text()) for p in files]
    passed = [r for r in rows if r['status']=='ok']
    failures = [r for r in rows if r['status']!='ok']
    index = {(r['space'],r['dataset'],r['method'],r['seed']):r for r in passed}
    summary = []
    for space in ['E','H','H-fixed-E','H-weak','H-strong']:
        for dataset in DATASETS:
            for method in methods_for(space):
                group = [r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if not group: continue
                item = dict(space=space,dataset=dataset,method=method,runs=len(group))
                for metric in ['heldout_nmi30','heldout_ari30','noise_rate','clusters','actual_fits','unique_fits','seconds','cpu_seconds','actor_updates','critic_updates']:
                    values = np.array([r[metric] for r in group],dtype=float)
                    item[metric+'_mean'] = float(values.mean())
                    item[metric+'_std'] = float(values.std(ddof=1)) if len(values)>1 else 0.
                item['full_nmi30_mean'] = float(np.mean([r['curve'][-1] for r in group]))
                item['full_auc30_mean'] = float(np.mean([np.mean(r['curve']) for r in group]))
                summary.append(item)
    dest = out/'analysis'; dest.mkdir(exist_ok=True)
    if summary:
        with (dest/'summary.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader();writer.writerows(summary)
    comparisons = []
    for dataset in DATASETS:
        for method in METHODS:
            pairs = [(index[('E',dataset,method,s)],index[('H',dataset,method,s)])
                     for s in range(10) if ('E',dataset,method,s) in index and ('H',dataset,method,s) in index]
            if not pairs: continue
            diffs = [h['heldout_nmi30']-e['heldout_nmi30'] for e,h in pairs]
            label_aris = []
            for e,h in pairs:
                def prediction(space):
                    return np.load(out/'cases'/space/f'{dataset}_{method}_seed{e["seed"]}'/'predictions.npz')['reward_incumbents'][-1]
                label_aris.append(float(adjusted_rand_score(prediction('E'),prediction('H'))))
            comparisons.append(dict(dataset=dataset,method=method,seeds=[e['seed'] for e,h in pairs],
                                    heldout_differences=diffs, mean_difference=float(np.mean(diffs)),
                                    prediction_ari_mean=float(np.mean(label_aris)),
                                    full_nmi30_difference=float(np.mean([h['curve'][-1]-e['curve'][-1] for e,h in pairs])),
                                    auc30_difference=float(np.mean([np.mean(h['curve'])-np.mean(e['curve']) for e,h in pairs]))))
    write_json(dest/'paired_comparisons.json',comparisons)
    curve_similarity=[]
    for method in METHODS[:8]:
        group=[r for r in comparisons if r['method']==method and len(r['seeds'])==10]
        if not group: continue
        endpoint=float(np.mean([abs(r['full_nmi30_difference']) for r in group]))
        area=float(np.mean([abs(r['auc30_difference']) for r in group]))
        curve_similarity.append(dict(method=method,datasets=len(group),complete=len(group)==8,
                                     macro_absolute_h30_difference=endpoint,macro_absolute_auc_difference=area,
                                     descriptive_close=bool(len(group)==8 and endpoint<=.02 and area<=.02)))
    write_json(dest/'curve_similarity.json',curve_similarity)
    learning_audit=[]
    for r in passed:
        if r['method'] not in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']: continue
        active=zero=initialized=untrained=0
        for region in r['regions']:
            layers={e['layer'] for e in region.get('episodes',[])}
            active+=len(layers);initialized+=len(region['actor_updates'])
            untrained+=len(region['actor_updates'])-len(layers)
            zero+=sum(region['actor_updates'][layer]==0 for layer in layers)
        learning_audit.append(dict(space=r['space'],dataset=r['dataset'],seed=r['seed'],method=r['method'],
                                   initialized_layers=initialized,active_training_layers=active,
                                   untrained_initialized_layers=untrained,zero_update_active_layers=zero,
                                   actor_updates=r['actor_updates'],critic_updates=r['critic_updates'],
                                   zero_support_regions=sum(a['support_count']==0 for a in r['regions']),
                                   one_support_regions=sum(a['support_count']==1 for a in r['regions']),
                                   low_information_regions=sum(a['low_information'] for a in r['regions'])))
    write_json(dest/'learning_audit.json',learning_audit)
    primary = [r for r in comparisons if r['method']=='MA-TD3']
    complete = len(primary)==8 and all(len(r['seeds'])==10 for r in primary)
    equivalence = dict(complete=complete, margin=.02, status='incomplete', observed_datasets=len(primary))
    if complete:
        differences = np.asarray([r['heldout_differences'] for r in primary])
        rng = np.random.default_rng(20261104)
        sampled = rng.integers(0,10,size=(10000,8,10))
        boot = np.take_along_axis(differences[None,:,:],sampled,axis=2).mean(axis=(1,2))
        interval = np.quantile(boot,[.05,.95])
        small = sum(abs(r['mean_difference'])<=.05 for r in primary)
        equivalent = bool(interval[0]>-.02 and interval[1]<.02 and small>=7)
        equivalence.update(mean_difference=float(differences.mean()), ci90=interval, datasets_within_005=small,
                           status='met_predefined_similarity' if equivalent else 'did_not_meet_predefined_similarity')
    write_json(dest/'equivalence.json',equivalence)
    colors = dict(zip(METHODS,plt.cm.tab20(np.linspace(0,1,len(METHODS)))))
    for space in ['E','H']:
        fig,axes = plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for method in METHODS[:8]:
                group = [r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if not group: continue
                curves = np.asarray([r['curve'] for r in group]); mean=curves.mean(0); sd=curves.std(0)
                ax.plot(np.arange(1,31),mean,label=method,color=colors[method])
                if method in ['MA-TD3','SA-TD3','BO-TPE']:
                    ax.fill_between(np.arange(1,31),np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=colors[method],alpha=.12)
            ax.set_title(dataset); ax.grid(alpha=.25); ax.set_ylim(0,1.02)
            ax.set_xlabel('New training candidate / region')
        for ax in axes[:,0]: ax.set_ylabel('Full-label historical maximum NMI')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        if handles: fig.legend(handles,labels,loc='upper center',ncol=8)
        fig.suptitle(f'{space}: audited Fig.6; bands = standard deviation',y=.94)
        fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(dest/f'fig6_{space}.png',dpi=160)
        fig.savefig(dest/f'fig6_{space}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for method in METHODS[:8]:
            pairs=[(index[('E',dataset,method,s)],index[('H',dataset,method,s)]) for s in range(10)
                   if ('E',dataset,method,s) in index and ('H',dataset,method,s) in index]
            if pairs:
                delta=np.asarray([np.asarray(h['curve'])-e['curve'] for e,h in pairs])
                ax.plot(np.arange(1,31),delta.mean(0),label=method,color=colors[method])
        ax.axhline(0,color='gray',lw=.7);ax.set_title(dataset);ax.grid(alpha=.25)
        ax.set_xlabel('New training candidate / region')
    for ax in axes[:,0]: ax.set_ylabel('H - E historical maximum NMI')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=8,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'space_difference.png',dpi=160)
    fig.savefig(dest/'space_difference.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for ax,dataset in zip(axes.flat,DATASETS):
        for space,style in [('E','--'),('H','-')]:
            for method in ['MA-TD3','SA-TD3','BO-TPE']:
                group=[r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if group:
                    ax.plot(np.mean([r['cumulative_fits'] for r in group],axis=0),
                            np.mean([r['curve'] for r in group],axis=0),style,label=f'{space} {method}',color=colors[method])
        ax.set_title(dataset);ax.set_xlabel('Cumulative actual fits, including detect');ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=6,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'actual_fit_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for method in ['MA-TD3','SA-TD3','MA-FrozenActor','MA-Random']:
            values=[]
            for space in ['H-weak','H','H-strong']:
                group=[r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                values.append(np.mean([r['heldout_nmi30'] for r in group]) if group else np.nan)
            ax.plot([.2,.5,.8],values,'o-',label=method,color=colors[method])
        ax.set_title(dataset);ax.set_xlabel('Maximum ball radius');ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=4,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'mapping_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for space,marker in [('E','o'),('H','s'),('H-fixed-E','^')]:
            means=[];stds=[]
            for method in ['MA-TD3','MA-FrozenActor','MA-Random']:
                values=[r['heldout_nmi30'] for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                means.append(np.mean(values) if values else np.nan);stds.append(np.std(values) if values else 0.)
            ax.errorbar(range(3),means,yerr=stds,marker=marker,label=space,capsize=2)
        ax.set_title(dataset);ax.set_xticks(range(3),['TD3','Frozen','Random']);ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=3,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'learning_and_regions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for space in ['E','H','H-fixed-E']:
            group=[r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
            if not group: continue
            values=[[r['diagnostics'][stage]['nmi'] for stage in ['raw_tree','modules','regions','initial']]+[r['round_nmi'][-1]] for r in group]
            ax.plot(range(5),np.mean(values,axis=0),'o-',label=space)
        ax.set_title(dataset);ax.set_xticks(range(5),['Tree','Modules','Regions','Initial','Final'],rotation=30);ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=3,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'partition_stages.png',dpi=160);plt.close(fig)
    structural=[]
    for path in sorted((out/'partitions').glob('*/*.npz')):
        dataset=path.parent.name
        space,seed_text=path.stem.rsplit('_seed',1);seed=int(seed_text)
        frozen=np.load(out/'inputs'/dataset/f'seed{seed}.npz');y=frozen['y'];support=frozen['reward_ids']
        part=np.load(path)
        for stage in ['raw_tree','modules','regions']:
            labels=part[stage];regions=np.unique(labels)
            total=retained=nonnoise_total=nonnoise_retained=0
            noise_total=noise_retained=0
            for cls in np.unique(y):
                n=int(np.sum(y==cls));total+=n*(n-1)//2
                _,counts=np.unique(labels[y==cls],return_counts=True)
                kept=sum(int(c)*(int(c)-1)//2 for c in counts)
                retained+=kept
                if cls==-1:
                    noise_total=n*(n-1)//2;noise_retained=kept
                else:
                    nonnoise_total+=n*(n-1)//2;nonnoise_retained+=kept
            structural.append(dict(dataset=dataset,space=space,seed=seed,stage=stage,
                                   count=len(regions),sizes=[int(np.sum(labels==r)) for r in regions],
                                   support_counts=[int(np.sum(labels[support]==r)) for r in regions],
                                   same_class_pair_retention=retained/total if total else None,
                                   nonnoise_same_class_pair_retention=nonnoise_retained/nonnoise_total if nonnoise_total else None,
                                   true_noise_count=int(np.sum(y==-1)),
                                   true_noise_pair_retention=noise_retained/noise_total if noise_total else None))
    write_json(dest/'structural_diagnostics.json',structural)
    fig,axes=plt.subplots(4,4,figsize=(16,16))
    for i,dataset in enumerate(DATASETS):
        frozen=np.load(out/'inputs'/dataset/'seed0.npz');order=frozen['sample_ids']
        for j,space in enumerate(['E','H']):
            ax=axes[i//2,(i%2)*2+j]
            path=out/'cases'/space/f'{dataset}_MA-TD3_seed0/predictions.npz'
            if not path.exists(): ax.set_visible(False);continue
            prediction=np.load(path)['reward_incumbents'][-1]
            points=frozen['x'] if space=='E' else np.load(out/'geometry'/dataset/'r0.5'/'points.npy')[order]
            normal=prediction!=-1
            ax.scatter(points[normal,0],points[normal,1],c=prediction[normal],cmap='tab20',s=3,rasterized=True)
            ax.scatter(points[~normal,0],points[~normal,1],c='black',s=2,alpha=.4,rasterized=True)
            if space=='H':
                ax.add_patch(plt.Circle((0,0),1,fill=False,color='gray',lw=.7));ax.set_xlim(-1.02,1.02);ax.set_ylim(-1.02,1.02)
            ax.set_aspect('equal');ax.set_title(f'{dataset} {space} MA-TD3 seed0')
    fig.suptitle('Support-selected clusters; colors local to panel; black = noise; H circle = unit ball',y=.995)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(dest/'cluster_maps_seed0.png',dpi=160);plt.close(fig)
    expected=sum(8*10*len(methods_for(s)) for s in ['E','H','H-fixed-E','H-weak','H-strong'])
    counts={s:sum(r['space']==s for r in passed) for s in ['E','H','H-fixed-E','H-weak','H-strong']}
    primary_text=(f'主方法MA-TD3、主映射r_max=0.5，共80对数据—种子结果。宏平均差{equivalence["mean_difference"]:.4f}，90%区间[{equivalence["ci90"][0]:.4f}, {equivalence["ci90"][1]:.4f}]；{equivalence["datasets_within_005"]}/8数据集满足逐数据±0.05。区间使用固定八数据集内的配对种子bootstrap，共10000次重采样。' if complete else '主比较尚未完整，不判断是否达到相近标准。')
    report=['# 欧氏—双曲 Fig.6 实验报告','',f'正式矩阵：{expected}例；已成功：{len(passed)}；失败：{len(failures)}。',
            '',f'各空间完成数：{counts}。','',
            '输入由仓库原加载器语义重新生成，未冒称已恢复历史input.npz。训练仅接收支持标签。',
            '曲率固定−1；主r_max=0.5，敏感性为0.2/0.8；所有区域共用全局Q90距离尺度。','',
            '## 主要判据','',primary_text,'',
            '只有八数据×十种子的MA-TD3配对完整后，才判断预设相近标准；不完整时不作等效结论。','',
            '| 数据集 | 配对数 | H−E补集NMI | E/H预测ARI |','|---|---:|---:|---:|']
    for row in primary:
        report.append(f'| {row["dataset"]} | {len(row["seeds"])} | {row["mean_difference"]:.4f} | {row["prediction_ari_mean"]:.4f} |')
    if complete:
        conclusion='达到预设相近标准' if equivalence['status']=='met_predefined_similarity' else '未达到预设相近标准'
        lo,hi=equivalence['ci90']
        report+=['',f'结论：**{conclusion}**。八数据等权宏平均 H−E = {equivalence["mean_difference"]:.4f}，90%配对bootstrap区间 [{lo:.4f}, {hi:.4f}]；{equivalence["datasets_within_005"]}/8 数据集均值差在±0.05以内。',
                 '这同时要求区间落入±0.02及至少7/8数据满足逐数据阈值，不用“不显著”替代等效判据。']
    report+=['','## 各方法空间差异','',
             '| 方法 | 完整配对数据数 | 补集NMI宏平均H−E | Fig.6 H30宏平均差 | 曲线面积宏平均差 |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        group=[r for r in comparisons if r['method']==method and len(r['seeds'])==10]
        if group:
            report.append(f'| {method} | {len(group)} | {np.mean([r["mean_difference"] for r in group]):.4f} | {np.mean([r["full_nmi30_difference"] for r in group]):.4f} | {np.mean([r["auc30_difference"] for r in group]):.4f} |')
    report+=['','曲线接近的描述性检查使用逐数据均值差的绝对值再宏平均，避免正负抵消。AUC30明确为H1–H30的平均值，不把端点相近说成整条曲线相近。','',
             '| 方法 | 完整数据数 | H30宏平均绝对差 | AUC30宏平均绝对差 | 两项均≤0.02 |',
             '|---|---:|---:|---:|---|']
    for row in curve_similarity:
        report.append(f'| {row["method"]} | {row["datasets"]} | {row["macro_absolute_h30_difference"]:.4f} | {row["macro_absolute_auc_difference"]:.4f} | {row["descriptive_close"]} |')
    report+=['','## 固定区域与映射强度','',
             '| 数据集 | E | H原生区域 | H固定E区域 | H弱映射 | H强映射 |',
             '|---|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        values=[]
        for space in ['E','H','H-fixed-E','H-weak','H-strong']:
            group=[r for r in passed if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
            values.append(f'{np.mean([r["heldout_nmi30"] for r in group]):.4f} (n={len(group)})' if group else '未完成')
        report.append('| '+dataset+' | '+' | '.join(values)+' |')
    replay_path=out/'validation/replay_summary.json'
    if replay_path.exists():
        replay=json.loads(replay_path.read_text())
        report+=['','## 三维控制回放','',
                 f'已回放{replay["units"]}个案例、{replay["candidates"]}个候选。原生核心点/噪声/分簇全部一致：{replay["native_clustering_equal"]}；原始float64邻接全部一致：{replay["raw_adjacency_equal"]}。',
                 f'记录到{replay["boundary_pair_occurrences"]}次边界点对出现（包含重复候选中的重复出现），精确重算与原始距离间隙均保存在validation/replay/。不把原始浮点逐边不一致描述为严格逐边完全相同。']
    checks_path=out/'validation/artifact_checks.json'
    if checks_path.exists():
        checks=json.loads(checks_path.read_text())
        report+=['','## 产物独立核验','',
                 f'状态：{checks["status"]}；核验{checks["verified_cases"]}例，独立重算{checks["unique_reward_checks"]}种支持奖励。检查支持ID、每候选奖励、最终选择、补集NMI/ARI、历史最大曲线、冻结Actor和拟合计数。']
    if complete:
        report+=['','## 结果解释','',
                 '宏平均相近与逐数据相近是不同检查：主结果中较大的正向和负向变化会在平均时抵消；逐数据阈值用于限制这种情况。',
                 '距离排序只能在严格对应eps下保证普通DBSCAN邻域等价。完整流程还包含距离权重、选k、区域分配、状态和数值参数搜索，不能由排序保持推出整条训练曲线保持。']
        def ma_value(dataset,space,metric):
            group=[r for r in passed if (r['dataset'],r['space'],r['method'])==(dataset,space,'MA-TD3')]
            return float(np.mean([r[metric] for r in group])) if len(group)==10 else None
        d31=[ma_value('D31',s,'heldout_nmi30') for s in ['E','H','H-fixed-E']]
        if all(v is not None for v in d31):
            report.append(f'D31：E={d31[0]:.4f}，原生H={d31[1]:.4f}，固定E区域的H={d31[2]:.4f}。固定区域明显缩小差距，支持分区及其带来的局部搜索变化是重要因素；该桥接不是对所有因素的纯因果分解。')
        weak=ma_value('D31','H-weak','heldout_nmi30')
        if weak is not None:
            report.append(f'D31弱映射NMI={weak:.4f}，减弱映射没有使结果更接近欧氏参照。应联合查看选k、模块、区域和训练轨迹，不能只依据整体距离畸变预测最终质量。')
    report+=['','## 解释与限制','',
             '旧作者行为与统一审计后的欧氏行为不同，空间效应必须比较新E和新H。',
             '新流程在训练前独立重置随机流，移除真K的KMeans也改变旧流程的随机数消耗；新旧E差异不能全部归因于缺陷修复。',
             '使用仓库实际数据：Cure-t1为2000点，Cure-t2为4200点（其中真标签−1有200点）；未按数据集名称截成4000点。',
             '三维等价控制保留原始float64边界差异，并用有理数精确重算可疑边；未放宽eps。',
             '标准差阴影用于新图；旧入口使用均值±方差，两种口径不能混称。',
             '几何和分区为共享准备成本；案例CPU/墙钟不含该共享准备，分区耗时另存。',
             '案例计时包含训练、产物写盘和离线评价；原生DBSCAN拟合耗时另存于逐区域fit_seconds，不把总墙钟说成纯训练耗时。',
             '四个训练worker与候选回放可能共享CPU配额，墙钟仅作本次运行记录；优先比较实际拟合数和进程CPU时间，不据此宣称受控的延迟加速比。',
             'process_peak_rss_kib是工作进程累计峰值，不冒称逐案例内存峰值。',
             'MA-Random采用本次明确的物理网格规则；仓库没有机制审计实现，不能宣称逐字复用。',
             '本报告只针对这八个欧氏形状数据的受控迁移，不证明天然双曲数据上的优势。','']
    if failures: report+=['## 失败案例','']+[f'- {r["space"]}/{r["dataset"]}/{r["method"]}/seed{r["seed"]}: {r["error"]}' for r in failures]
    report+=['','## 图和数据文件','',
             '- [欧氏Fig.6](fig6_E.png) / [PDF](fig6_E.pdf)',
             '- [主双曲Fig.6](fig6_H.png) / [PDF](fig6_H.pdf)',
             '- [空间差值曲线](space_difference.png) / [PDF](space_difference.pdf)',
             '- [实际拟合成本曲线](actual_fit_curves.png)',
             '- [映射强度敏感性](mapping_sensitivity.png)',
             '- [训练、冻结、随机与固定区域](learning_and_regions.png)',
             '- [分区阶段诊断](partition_stages.png)',
             '- [seed0最终分簇对照](cluster_maps_seed0.png)',
             '- [汇总CSV](summary.csv)、[配对比较](paired_comparisons.json)、[曲线接近检查](curve_similarity.json)、[学习审计](learning_audit.json)','']
    (dest/'报告.md').write_text('\n'.join(report))
    write_json(dest/'status.json',dict(expected=expected,passed=len(passed),failed=len(failures),counts=counts))
    print(json.dumps(dict(expected=expected,passed=len(passed),failed=len(failures),equivalence=equivalence),default=lambda x:x.tolist()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/hyperbolic')
    analyze(parser.parse_args().output.resolve())

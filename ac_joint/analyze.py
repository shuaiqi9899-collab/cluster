"""Offline analysis for joint AC search, including selected linkage frequency."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from hyperbolic.prepare import ROOT,DATASETS,write_json
from .experiment import METHODS,SPACES,methods_for
from .trees import LINKAGES


def analyze(out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(p.read_text()) for p in sorted((out/'cases').glob('*/*/result.json'))]
    valid=[r for r in rows if r['status']=='ok'];failures=[r for r in rows if r['status']!='ok']
    dest=out/'analysis';dest.mkdir(exist_ok=True)
    index={(r['space'],r['dataset'],r['method'],r['seed']):r for r in valid}
    summary=[]
    for space in SPACES:
        for dataset in DATASETS:
            for method in methods_for(space):
                group=[r for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if not group:continue
                item=dict(space=space,dataset=dataset,method=method,runs=len(group))
                for metric in ['heldout_nmi30','heldout_ari30','full_training_heldout_nmi','clusters','singleton_fraction',
                               'largest_cluster_fraction','agents','all_training_requests','total_queries','actual_cuts',
                               'unique_parameter_pairs','unique_clusterings','actor_updates','critic_updates','seconds','cpu_seconds']:
                    values=[r[metric] for r in group];item[metric+'_mean']=float(np.mean(values));item[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                item['full_nmi30_mean']=float(np.mean([r['curve'][-1] for r in group]))
                item['auc30_mean']=float(np.mean([np.mean(r['curve']) for r in group]))
                summary.append(item)
    if summary:
        with (dest/'summary.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    pairs=[]
    for dataset in DATASETS:
        for method in METHODS:
            group=[(index['E',dataset,method,s],index['H',dataset,method,s]) for s in range(10)
                   if ('E',dataset,method,s) in index and ('H',dataset,method,s) in index]
            if not group:continue
            agreements=[]
            for e,h in group:
                def labels(space):return np.load(out/'cases'/space/f'{dataset}_{method}_seed{e["seed"]}'/'predictions.npz')['reward_incumbents'][-1]
                agreements.append(float(adjusted_rand_score(labels('E'),labels('H'))))
            pairs.append(dict(dataset=dataset,method=method,seeds=[e['seed'] for e,h in group],
                              differences=[h['heldout_nmi30']-e['heldout_nmi30'] for e,h in group],
                              mean_difference=float(np.mean([h['heldout_nmi30']-e['heldout_nmi30'] for e,h in group])),
                              mean_prediction_ari=float(np.mean(agreements)),
                              h30_difference=float(np.mean([h['curve'][-1]-e['curve'][-1] for e,h in group])),
                              auc_difference=float(np.mean([np.mean(h['curve'])-np.mean(e['curve']) for e,h in group]))))
    write_json(dest/'paired_comparisons.json',pairs)
    primary=[r for r in pairs if r['method']=='MA-TD3'];complete=len(primary)==8 and all(len(r['seeds'])==10 for r in primary)
    equivalence=dict(complete=complete,status='incomplete',margin=.02)
    if complete:
        diffs=np.asarray([r['differences'] for r in primary]);rng=np.random.default_rng(20261105)
        sampled=rng.integers(0,10,size=(10000,8,10))
        bootstrap=np.take_along_axis(diffs[None],sampled,axis=2).mean(axis=(1,2))
        ci=np.quantile(bootstrap,[.05,.95]);within=sum(abs(r['mean_difference'])<=.05 for r in primary)
        close=bool(ci[0]>-.02 and ci[1]<.02 and within>=7)
        equivalence.update(mean_difference=float(diffs.mean()),ci90=ci,datasets_within_005=within,
                           status='met_predefined_similarity' if close else 'did_not_meet_predefined_similarity')
    write_json(dest/'equivalence.json',equivalence)
    frequencies=[]
    for r in valid:
        count=collections.Counter(r['selected_linkages'])
        sample_count=collections.Counter()
        for linkage,region in zip(r['selected_linkages'],r['regions']):sample_count[linkage]+=region['size']
        frequencies.append(dict(space=r['space'],dataset=r['dataset'],method=r['method'],seed=r['seed'],
                                 agents=r['agents'],counts={l:count[l] for l in LINKAGES},
                                 sample_counts={l:sample_count[l] for l in LINKAGES}))
    write_json(dest/'linkage_choices.json',frequencies)
    colors=dict(zip(METHODS,plt.cm.tab20(np.linspace(0,1,len(METHODS)))))
    displayed=METHODS[:8]+['Grid-30']
    for space in ['E','H']:
        fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for method in displayed:
                group=[r for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if not group:continue
                values=np.asarray([r['curve'] for r in group]);mean=values.mean(0);sd=values.std(0)
                ax.plot(np.arange(1,31),mean,label=method,color=colors[method],linestyle='--' if method in ['SA-TD3','Grid-30'] else '-',
                        marker='o' if method=='MA-TD3' else None,markevery=5,markersize=3)
                if method in ['MA-TD3','SA-TD3','BO-TPE']:ax.fill_between(np.arange(1,31),np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.1,color=colors[method])
            ax.set_title(dataset);ax.set_ylim(0,1.02);ax.grid(alpha=.25);ax.set_xlabel('Training candidate requests / region')
        for ax in axes[:,0]:ax.set_ylabel('Full-label historical maximum NMI')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,ncol=9,loc='upper center')
        fig.suptitle(f'{space}: joint linkage + threshold AC; first 30 requests incl. warmups/cache hits',y=.94)
        fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(dest/f'fig6_AC_{space}.png',dpi=160);fig.savefig(dest/f'fig6_AC_{space}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for method in displayed:
            group=[(index['E',dataset,method,s],index['H',dataset,method,s]) for s in range(10)
                   if ('E',dataset,method,s) in index and ('H',dataset,method,s) in index]
            if group:ax.plot(range(1,31),np.mean([np.asarray(h['curve'])-e['curve'] for e,h in group],axis=0),label=method,color=colors[method])
        ax.set_title(dataset);ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,ncol=9,loc='upper center')
    for ax in axes[:,0]:ax.set_ylabel('H - E historical maximum NMI')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'space_difference.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for method in ['MA-TD3','SA-TD3','MA-FrozenActor','MA-Random']:
            means=[]
            for space in ['H-weak','H','H-strong']:
                group=[r['heldout_nmi30'] for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                means.append(np.mean(group) if group else np.nan)
            ax.plot([.2,.5,.8],means,'o-',label=method,color=colors[method])
        ax.set_title(dataset);ax.set_xlabel('Maximum Poincare radius');ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,ncol=4,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'mapping_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for space,style in [('E','--'),('H','-'),('H-fixed-E',':')]:
            values=[]
            for method in ['MA-TD3','MA-FrozenActor','MA-Random']:
                group=[r['heldout_nmi30'] for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                values.append(np.mean(group) if group else np.nan)
            ax.plot(range(3),values,style,marker='o',label=space)
        ax.set_title(dataset);ax.set_xticks(range(3),['TD3','Frozen','Random']);ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,ncol=3,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'learning_and_regions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for i,space in enumerate(['E','H']):
            rows_f=[r for r in frequencies if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
            if not rows_f:continue
            counts=np.array([sum(r['counts'][l] for r in rows_f) for l in LINKAGES],dtype=float);counts/=counts.sum()
            ax.bar(np.arange(4)+(i-.5)*.35,counts,width=.35,label=space)
        ax.set_title(dataset);ax.set_xticks(range(4),LINKAGES,rotation=30)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,ncol=2,loc='upper center')
    for ax in axes[:,0]:ax.set_ylabel('Fraction of selected local linkage types')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'selected_linkages.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for ax,dataset in zip(axes.flat,DATASETS):
        for space,style in [('E','--'),('H','-')]:
            for method in ['MA-TD3','SA-TD3','Grid-30']:
                group=[r for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,method)]
                if group:ax.plot(np.mean([r['total_queries_at_display'] for r in group],axis=0),np.mean([r['curve'] for r in group],axis=0),style,label=f'{space} {method}',color=colors[method])
        ax.set_title(dataset);ax.set_xlabel('All cut queries incl. detect/cache hits');ax.grid(alpha=.25)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,ncol=6,loc='upper center')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'query_cost_curves.png',dpi=160);plt.close(fig)
    counts={s:sum(r['space']==s for r in valid) for s in SPACES}
    report=['# 四链接＋距离阈值 AC：欧氏—双曲实验','',f'正式矩阵2800例；已成功{len(valid)}，失败{len(failures)}。',
            '',f'完成数：{counts}。','',
            '## 方法与预算','',
            '每个agent搜索链接类型single/complete/average/ward和距离阈值，不搜索K。主后端输出完整分区，无DBSCAN式噪声拒绝。',
            '双曲Ward为Fréchet平方测地距离和的合并增量，合并高度sqrt(2Δ)。CAT(0)方差下界只用于筛选候选，实际增量通过原生Fréchet目标求解；不是切空间Ward或欧氏递推。',
            '阈值采用闭区间切分，必要时用子树最大高度处理倒置；四类链接分别以实际全局/区域树最大有效高度归一化，各区域共用尺度。',
            '用户选择完整训练：三层独立网络，最多14局/层、每局最多30步、batch16；仅回顾性展示前30次训练请求，包含四类初始探针及缓存命中。完整训练和detect的额外成本全部保留。',
            '基线总预算30次，四类探针计入预算。PSO/GA/DE为四个分类外层＋一维阈值内层，pop4、总配额8/8/7/7，按seed轮换；没有把链接类别当有序连续整数。',
            '状态中的当前预测簇数来自候选输出，未使用真实K；训练仅持有20%支持标签，补集和全标签仅离线评价。','',
            '## 主要空间比较','']
    if complete:
        lo,hi=equivalence['ci90'];conclusion='达到' if equivalence['status']=='met_predefined_similarity' else '未达到'
        report.append(f'主MA-TD3：H−E补集NMI宏平均差{equivalence["mean_difference"]:.4f}，90%配对bootstrap区间[{lo:.4f}, {hi:.4f}]；{equivalence["datasets_within_005"]}/8数据集差值在±0.05内，**{conclusion}预设相近标准**。')
    else:report.append('主要配对尚不完整，不作相近判定。')
    report+=['','| 数据集 | E | H | H固定E区域 | H弱 | H强 | E/H预测ARI |','|---|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        values=[]
        for space in SPACES:
            group=[r['heldout_nmi30'] for r in valid if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
            values.append(f'{np.mean(group):.4f} (n={len(group)})' if group else '未完成')
        pair=next((r for r in primary if r['dataset']==dataset),None)
        report.append('| '+dataset+' | '+' | '.join(values)+' | '+(f'{pair["mean_prediction_ari"]:.4f}' if pair else '未完成')+' |')
    report+=['','## 各搜索方法','',
             '| 方法 | 完整配对数据数 | 补集NMI宏平均H−E | H30宏平均绝对差 | AUC30宏平均绝对差 |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        group=[r for r in pairs if r['method']==method and len(r['seeds'])==10]
        if group:report.append(f'| {method} | {len(group)} | {np.mean([r["mean_difference"] for r in group]):.4f} | {np.mean([abs(r["h30_difference"]) for r in group]):.4f} | {np.mean([abs(r["auc_difference"]) for r in group]):.4f} |')
    report+=['','## 完整训练与Actor冻结','',
             '完整训练列使用全部训练请求中由支持奖励选出的incumbent，与前30请求结果分开；其查询成本更高，不能与30请求基线直接当作等预算比较。下表按八数据等权，每数据十种子。','',
             '| 空间 | 方法 | 前30补集NMI | 全训练补集NMI | 全部查询均值 | Actor更新均值 |',
             '|---|---|---:|---:|---:|---:|']
    for space in ['E','H']:
        for method in ['MA-TD3','MA-FrozenActor','SA-TD3','SA-FrozenActor']:
            group=[r for r in valid if r['space']==space and r['method']==method]
            if len(group)==80:
                report.append(f'| {space} | {method} | {np.mean([r["heldout_nmi30"] for r in group]):.4f} | {np.mean([r["full_training_heldout_nmi"] for r in group]):.4f} | {np.mean([r["total_queries"] for r in group]):.1f} | {np.mean([r["actor_updates"] for r in group]):.1f} |')
    report.append('本次均值未显示Actor更新带来稳定NMI优势：SA训练/冻结的前30指标一致，MA完整训练的冻结均值也略高。Actor影响轨迹和请求量，但不能把高NMI全部归因于强化学习。')
    report+=['','## 主方法的链接选择','',
             '以下按最终所选区域覆盖的样本数加权；图中另给区域计数比例。均为第30步支持奖励incumbent，不按真实K选择。','',
             '| 数据集 | 空间 | 单链接 | 全链接 | 平均链接 | Ward |','|---|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for space in ['E','H']:
            group=[r for r in frequencies if (r['space'],r['dataset'],r['method'])==(space,dataset,'MA-TD3')]
            if not group:continue
            counts=np.array([sum(r['sample_counts'][l] for r in group) for l in LINKAGES],dtype=float);counts/=counts.sum()
            report.append('| '+dataset+' | '+space+' | '+' | '.join(f'{v:.1%}' for v in counts)+' |')
    sweep_files=list((out/'sweeps').glob('*/*_seed*.json'))
    if sweep_files:
        sweeps=[json.loads(p.read_text()) for p in sweep_files]
        report+=['','## 完整阈值扫描诊断','',f'已完成{len(sweeps)}个数据—空间—种子组，分类及所有有效高度均纳入支持目标扫描；仅在支持预测变化时重新计算NMI。',
                 '该扫描只给出每个局部支持奖励的完整搜索参照，不是补集或合并后全局NMI上界。支持奖励并列时按固定类别顺序及最早阈值选择。']
        for space in ['E','H']:
            vals=[r['results']['global']['heldout_nmi'] for r in sweeps if r['space']==space]
            if vals:report.append(f'{space}全局扫描参照：{len(vals)}组，补集NMI均值{np.mean(vals):.4f}（数据未齐时不作完整宏平均解释）。')
        report+=['','| 数据集 | E全局扫描 | H全局扫描 | E区域扫描 | H区域扫描 | H固定E区域扫描 |',
                 '|---|---:|---:|---:|---:|---:|']
        for dataset in DATASETS:
            values=[]
            for space,scope in [('E','global'),('H','global'),('E','native_regions'),('H','native_regions'),('H','fixed_E_regions')]:
                group=[r['results'][scope]['heldout_nmi'] for r in sweeps if r['dataset']==dataset and r['space']==space]
                values.append(f'{np.mean(group):.4f} (n={len(group)})' if group else '未完成')
            report.append('| '+dataset+' | '+' | '.join(values)+' |')
    check_path=out/'validation/artifact_checks.json'
    if check_path.exists():
        check=json.loads(check_path.read_text())
        report+=['','## 独立核验','',f'核验状态{check["status"]}；已验证{check["verified_cases"]}例、{check["independent_cut_checks"]}种链接/分区组合及{check["independent_reward_checks"]}次独立支持奖励。独立切树使用SciPy fcluster闭区间距离切分。']
    ward_path=out/'validation/ward_objectives.json'
    if ward_path.exists():
        ward=json.loads(ward_path.read_text())
        report.append(f'原生双曲Ward另以独立Karcher求解器核验{ward["native_ward_trees"]}棵树、{ward["checked_merge_objectives"]}个选定合并目标，状态{ward["status"]}。小规模逐步全候选最小增量核验在单元测试中执行。')
    report+=['','## 解释限制','',
             '这是已有二维欧氏形状数据的受控双曲迁移，不是天然双曲数据上的优势证明。',
             '四种链接的阈值不可直接横向比较，须连同链接类型和尺度解释。Ward合并高度还受簇大小影响。',
             '固定树上阈值决定切分和K，未使用K作为搜索变量；可达分区族仍受该树限制。',
             '区域独立聚类且不跨区域合并，最终K不小于区域数；错误预分区可限制后端恢复。',
             'AC没有噪声拒绝，Cure-t2的真实噪声仍计入离线指标；不能将预测噪声率0解释成更好。',
             '完整训练多于30个实际查询，前30请求曲线不是30请求硬训练预算。必须结合实际查询、切树数、更新数和完整成本判断效率。',
             '设置链接类型的动作允许重复当前类型，产生缓存命中；四类共同初始探针也可能已给出较高分数。因此前30曲线可较早变平，不能把30请求说成30次有效新分区探索。',
             '建树在方法和种子之间共享，单独记录；单例CPU/墙钟包含状态、训练、保存和评价，墙钟受并发影响。进程RSS是累计峰值。',
             '新AC图使用请求口径，旧DBSCAN图使用新训练缓存条目口径，不能直接把两图同一横坐标当成相同计算预算。',
             '双曲Ward为数值求解的原生目标，保留残差、成本误差界及合并审计，不宣称精确实数运算。','',
             '## 图与数据','',
             '- [欧氏AC Fig.6](fig6_AC_E.png) / [PDF](fig6_AC_E.pdf)',
             '- [双曲AC Fig.6](fig6_AC_H.png) / [PDF](fig6_AC_H.pdf)',
             '- [空间差值](space_difference.png)',
             '- [链接选择频率](selected_linkages.png)',
             '- [映射强度](mapping_sensitivity.png)',
             '- [冻结、随机、固定区域对照](learning_and_regions.png)',
             '- [实际查询曲线](query_cost_curves.png)',
             '- [汇总CSV](summary.csv)、[配对比较](paired_comparisons.json)、[链接选择](linkage_choices.json)','']
    (dest/'报告.md').write_text('\n'.join(report))
    write_json(dest/'status.json',dict(expected=2800,passed=len(valid),failed=len(failures),counts=counts))
    print(json.dumps(dict(expected=2800,passed=len(valid),failed=len(failures),equivalence=equivalence),default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/ac_joint');analyze(p.parse_args().output.resolve())

# 四链接与距离阈值联合搜索 AC

本模块新增欧氏/双曲凝聚聚类实验，保留上一轮 `hyperbolic/` 和原作者代码。使用现有 `/workspace/cluster` checkout，不创建额外Git worktree。

本次2800例正式实验及独立核验已完成。[完整报告](report/报告.md)、[欧氏图](report/fig6_AC_E.pdf)、[双曲图](report/fig6_AC_H.pdf)和汇总数据在 `report/`。逐例轨迹、权重、树库等大体积产物在忽略目录 `results/ac_joint/`；报告副本为本次冻结交付，后续新运行不会自动覆盖。

## 参数和几何

每个agent搜索 `(linkage,u)`，链接类型为single、complete、average、ward，实际距离阈值 `tau=u*S_linkage`。K来自切树输出，不是输入或搜索变量；状态中的预测K/n也不读取真实K。

- 单/全/平均链接使用相应欧氏或双曲点对距离的最小/最大/平均值。
- 欧氏Ward使用标准SciPy Ward。
- 双曲Ward定义 `W(C)=min_mu sum_i d_H(x_i,mu)^2`，合并代价 `Delta=W(A union B)-W(A)-W(B)`，高度 `sqrt(2*Delta)`。使用原生Fréchet均值，未采用切空间Ward或欧氏Lance–Williams递推。
- 每个数据集、空间和链接类型以实际全局/区域树最大有效根高度作为S；各区域共享，H原生与固定E区域共享同一S。尺度不使用标签。
- 切树采用闭区间 `effective_height <= tau`；effective_height是节点原始高度及所有后代高度的最大值，保留倒置统计。重复距离/零距离行为有测试。
- 不确定性标量的区域分组改为相邻差≤0.3的连通分组，与旧DBSCAN(min_samples=1)分组逐项核验等价；它是辅助分区步骤，不属于待搜索的主链接类型。

输入复用前次冻结的八数据集、seed0–9、20%支持ID，聚类树按原始样本ID规范排序。树只依赖点和几何，因此在相同点集上跨方法/打乱种子复用；区域集合不同则使用不同树。E/H输出均按原始样本ID排列。

## 双曲Ward算法与验证

Hadamard/CAT(0)方差不等式给出

`Delta(A,B) >= n_A*n_B/(n_A+n_B) * d_H(mean_A,mean_B)^2`。

这只用于惰性候选筛选。候选的真正代价仍通过合并点集的Fréchet目标求解。新均值使用常曲率黎曼Newton法和回溯，残差 `||mean(log_mu(x))||_mu <= 1e-9`。

对已求解均值，强凸性给出SSE误差界 `n*residual^2`；合并审计同时保留该界及float64相消误差余量。计算过的候选按保守代价上估计排序，合并选择精度由误差界限定，不能宣称精确实数排序。超出界的负增量会失败，界内舍入修正单独计数。

初始单点对使用排序的紧凑距离数组，新簇候选使用优先队列；淘汰过期候选并周期压缩。该算法不把中心距离下界当作实际Ward目标。已用小规模每一步全候选Fréchet代价枚举验证，另有持久化树的独立Karcher目标审计。

## 网络、请求和成本

用户选择：**完整训练，展示前30次训练候选请求，记录全部成本**。

- 每区域三层独立Actor/Critic，步长0.2/0.04/0.008，每层最多14局，每局最多30步，batch16，原TD3奖励回填和探索std2。
- 全局状态8维：u、两侧边界距离、预测K/n、链接类型one-hot4；局部状态4维：几何中心距离、大小、中心位置二维坐标图。
- 七动作：阈值减/增、设置四种链接、停止。切换类别保留u，实际tau随类别尺度换算。沿用原实现连续打分后argmax的动作代理，不把链接类别作为有序整数搜索。
- 状态的双曲中心用Fréchet均值，位置为固定logmap坐标图，距离除以前次全局Q90尺度。训练仅持有支持标签，完整标签只在离线评价使用。
- 四种链接各在u=.5初始化，按seed轮换顺序，四次都计入展示预算；不根据真实K初始化。
- 完整训练的检测和缓存请求不隐去。显示前30次训练请求，包含重复请求；不足持有末结果。该口径不同于旧DBSCAN图的前30个新缓存条目。
- AC全点分配，没有MinPts、噪声拒绝或小簇剔除。报告预测簇数、单点簇比例和最大簇比例，不能把噪声率0当作优势。
- 训练网络用等价的批量ragged attention计算替代Python逐样本循环，结构和参数层保持一致；有输出和梯度对照测试。

每次记录链接、u、实际tau、支持奖励、预测、缓存命中及实际切树数。建树按共享准备成本单列，不能将30次切树当成30次独立建树。CPU/墙钟含训练、状态、保存和评价；墙钟受并发影响，进程RSS为累计峰值。

## 方法与正式矩阵

12方法：MA-TD3、SA-TD3、Rand、BO-TPE、Anneal、PSO、GA、DE、MA-FrozenActor、SA-FrozenActor、MA-Random、Grid-30。

所有基线总预算30，含4个共同初始探针。BO/退火显式使用分类参数。PSO/GA/DE枚举四类别、各自搜索一维u，pop4，总配额8/8/7/7按seed轮换；属于本次明确的混合搜索重建设置。Grid-30沿同配额铺设阈值。冻结组只阻止Actor optimizer.step，其余回放、Critic和探索保留，并检查权重未变。

E/H各 `8*10*12=960`，固定E区域的H三种MA方法240，弱/强映射各四个核心方法320，总2800。主半径0.5，弱/强0.2/0.8；不挑选表现最好的映射替换主设置。

主要指标：第30步由支持奖励选择的预测，在支持补集上的NMI。预设相近标准沿用前次：八数据宏平均H−E的90%配对bootstrap区间完全在±0.02内，且至少7/8数据均值差在±0.05内。10000次数据集内种子重采样，仅解释这八个基准的运行随机性。

保存全标签历史最大曲线只作Fig.6兼容展示，不能作为部署选择。AUC30定义为H1–H30均值。另保存全训练支持incumbent，但其完整成本不同，不冒充30请求结果。

完整高度扫描是独立诊断：扫描四类树所有支持目标变化的平台，按支持NMI选取类型/阈值。它是各局部支持目标的穷举参照，**不是补集或合并后全局NMI上界**。主比较采用用户选定的完整训练口径，没有另加120次硬预算实验。

## 执行

已有环境 `.venv/bin/python`，依赖复用 `hyperbolic/requirements-lock.txt`；必要时运行 `bash hyperbolic/setup.sh`。无需新密钥或额外网络域。

```bash
cd /workspace/cluster
.venv/bin/python -m unittest ac_joint.test_trees -v
.venv/bin/python -u -m ac_joint.bootstrap --workers 2
.venv/bin/python -u -m ac_joint.prepare --workers 2
.venv/bin/python -m unittest ac_joint.test_search ac_joint.test_sweep -v
.venv/bin/python -u -m ac_joint.experiment --datasets Compound Pathbased D31 --spaces E H --seeds 0 --workers 2
.venv/bin/python -u -m ac_joint.experiment --workers 4
.venv/bin/python -u -m ac_joint.finish
```

树准备独立于搜索。bootstrap在缺少前次源产物时创建冻结输入/几何和SE模块，不执行DBSCAN训练；现有源产物只复用，不覆盖。源目录不完整且非空时应保留它并选择空的 `--source`，随后各命令使用相同路径。不能假设进程随快照恢复。先准备树再跑依赖树库的搜索测试。

成功案例可续跑跳过；失败案例保留并阻止静默覆盖。复用前核对训练源码与invocation哈希，行为变更使用新输出根目录。试运行只有训练源码和配置相同时可计入正式格。

`finish`等待2800例后依次执行Ward独立审计、完整阈值扫描、所有产物验证及分析。完整产物在 `results/ac_joint/`，初始失败试运行另存 `failed_smoke_v1/`，不纳入正式矩阵。

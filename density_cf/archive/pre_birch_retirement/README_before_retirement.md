# HDBSCAN / BIRCH 欧氏—双曲后端

在已有 DBSCAN 与 AC 实验之外新增两个后端，复用原有点ID、20%支持标签、SE区域和混合参数搜索器。已有实验目录保持不变。

| 后端 | 分类参数（四个设置动作） | 连续坐标u的实际含义 | 双曲实现 |
|---|---|---|---|
| HDBSCAN | min_samples = 1/5/10/20，局部截断到n−1 | 对数最小簇大小2…ceil(n_local/4)，取整 | 原生测地距离矩阵、互可达层次树、EOM |
| BIRCH | branching_factor = 10/25/50/100 | 对数CF半径δ/(10√N)…Dmax | Fréchet均值与测地RMS半径的CF扩展 |

均不输入或搜索真实K。HDBSCAN最小簇大小是样本数量，不是簇数量；BIRCH固定`n_clusters=None`。

HDBSCAN固定hdbscan 0.8.40，使用其generic/precomputed层次与提取私有API，已与公开HDBSCAN类逐参数核对。`min_samples`口径不含自身。固定EOM、alpha1、epsilon0、allow_single_cluster=False，噪声保持−1。

欧氏BIRCH为sklearn 1.2.2。双曲BIRCH保留成员ID以更新Fréchet均值/SSE，使用原生距离选入口、分裂及最终最近叶中心赋值；这是几何扩展，不宣称保有标准CF的紧凑内存界。相同seed的欧氏和双曲拟合使用相同冻结输入顺序。数值均值残差阈值1e−9，半径接受条件为严格浮点`R<=threshold`，近边界决策记录在缓存元数据。

参数域全部由无标签几何或局部样本数固定。δ取全局最小正距离，N为全局点数。极小半径的精确快捷路径基于CAT(0)方差不等式：含不同点的CF，SSE≥δ²/2，故R≥δ/√(2N)；低于该值只能合并重复坐标。普通半径不能重用其他阈值的CF树。

```bash
# 已有前两轮环境时，只新增这个依赖，不升级旧环境
uv pip install --python .venv/bin/python --no-deps -r density_cf/requirements.txt
.venv/bin/python -m pip check
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest density_cf.test_backends -v

# 全矩阵：每后端2800例，共5600例；支持断点续跑成功案例
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m density_cf.complete --workers 4
# complete依次完成实验、奖励/产物复核、独立后端核对与分析出图。
# results/density_cf/run_status.json记录当前阶段与完成数。
# 若仅需手动执行各阶段，也可使用下列命令：
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m density_cf.experiment --workers 4
.venv/bin/python -m density_cf.verify
.venv/bin/python -m density_cf.audit
.venv/bin/python -m density_cf.analyze

# 小规模核对
.venv/bin/python -m density_cf.experiment --datasets Pathbased --spaces E H --seeds 0 --workers 4
```

前提输入：`results/hyperbolic/`中的冻结样本、几何与`results/ac_joint/banks/`中的冻结区域ID。没有这些数据时，先按前两轮README准备。新结果单独存放于`results/density_cf/`；可审核小报告位于[report/报告.md](report/报告.md)。

如果尚无AC树，无需为了新后端先构造Ward等树：先执行`.venv/bin/python -m ac_joint.bootstrap --workers 2`生成冻结几何/SE区域，再运行`.venv/bin/python -m density_cf.prepare`创建轻量区域ID索引，实验命令加`--ac-source results/density_cf/region_inputs`即可。此索引只包含成员ID，不读取标签，也不构造任何AC树。

训练沿用用户已选口径：3层独立TD3，完整训练，回顾性展示每区域前30次训练请求；含4类初始探针、重复请求和缓存命中，完整训练/检测查询与网络更新另计。基线30次，PSO/GA/DE枚举分类参数后做一维u优化。支持NMI选参数；补集NMI/ARI只离线评价。`Fig.6`式全标签历史最大曲线只用于展示，另提供支持奖励incumbent的非单调补集曲线。

缓存与计数：编码坐标不同可能映射同一整数min_cluster_size，分别记录不同编码参数、不同物理参数与不同分区。层次树/预测可跨方法共享，但每次请求仍计入预算。每个案例保存完整查询、事件、候选预测、30步与全训练选参、策略权重。旧优化器内部的`linkage_id`仅是四类动作接口别名，公开参数记录使用实际后端名称。

为兼容既有预测文件格式，`actual_cuts_at_display`字段名保留，在本包中表示展示时的累计实际后端拟合/提取次数；HDBSCAN建互可达层次树的额外成本单列。新事件使用`actual_fits`。

完整矩阵：E/H各960例，H固定E区域240例、H弱/强各320例。MA使用冻结区域；SA和一般优化基线为全局。区域间不合并。HDBSCAN全噪声输出允许存在；噪声在状态中汇为一个token，在最终合并中仍保留−1。支持NMI把噪声视为一个拒绝标签，需同时检查拒绝率。

重复运行同一命令会跳过已有成功案例；失败案例保留并停止，排查后应使用新输出目录，避免混合协议。首次批量运行前将源码哈希写入invocations记录。墙钟受并发与缓存温度影响，不作为冷启动公平效率对比。

即使点对距离排序保持不变，HDBSCAN的EOM稳定性依赖逆距离，非线性单调变换仍可能改变选簇；BIRCH的均值和RMS半径也不只依赖排序。因此相近效果须实测，不能从排序不变直接推得。

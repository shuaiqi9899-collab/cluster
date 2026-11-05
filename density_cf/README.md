# HDBSCAN 欧氏—双曲实验

**BIRCH已按用户指示终止并淘汰。** 不再训练、续跑或纳入正式比较。已有538例与原报告只作归档，剩余2262例取消。继续保留的算法为DBSCAN、AC、HDBSCAN。

本包当前仅启用HDBSCAN。公共命令及程序接口拒绝BIRCH；调度、核验、分析与出图只覆盖HDBSCAN。原BIRCH数据仍在`results/density_cf/BIRCH/`，终止记录为`results/density_cf/retirement.json`。旧报告存于[archive/pre_birch_retirement](archive/pre_birch_retirement)，不再参与结论。

HDBSCAN已保存2800例十种子结果：E/H各960例，H固定E区域240例，H弱/强各320例。冻结八个数据集、seed0–9、20%支持标签及SE区域。搜索`min_samples∈{1,5,10,20}`（局部截断到n−1）和对数最小簇大小`2…max(2,ceil(n_local/4))`，不输入真实K。固定hdbscan0.8.40、EOM、alpha1、epsilon0、allow_single_cluster=False；原生测地距离用于双曲空间，噪声保持−1。

训练沿用完整三层TD3，最多14局/层、每局30步、batch16；回顾性展示每区域前30次训练请求，包含四类探针与缓存命中。全部训练/detect请求、物理参数、不同分区和网络更新成本另存。支持NMI选参数；补集NMI/ARI仅离线评价。Fig.6式全标签历史最大曲线仅为展示，另给支持奖励incumbent的补集曲线。

```bash
# 仅在需要安装依赖时执行，保持旧环境版本
uv pip install --python .venv/bin/python --no-deps -r density_cf/requirements.txt
.venv/bin/python -m pip check

# 当前范围应为HDBSCAN：2800/2800，BIRCH被排除
.venv/bin/python -m density_cf.complete --dry-run

# 仅补缺失的HDBSCAN案例，然后核验并更新图表
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m density_cf.complete --workers 4

# 手动执行各阶段
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m density_cf.experiment --backends HDBSCAN --workers 4
.venv/bin/python -m density_cf.verify
.venv/bin/python -m density_cf.audit
.venv/bin/python -m density_cf.analyze
```

正式图表、CSV与配对比较见[report/报告.md](report/报告.md)，进度见`results/density_cf/run_status.json`。未完成独立核验前，保存的训练完成数不代表已核验完成。

输入仍为`results/hyperbolic/`的冻结样本/几何和`results/ac_joint/banks/`的区域ID；旧DBSCAN与AC源码、结果、报告保持原样。没有AC树时，可以先运行`ac_joint.bootstrap --workers 2`生成几何/SE，再运行`density_cf.prepare`建立轻量区域索引；实验加`--ac-source results/density_cf/region_inputs`即可，不必构造Ward树。

为了保留历史来源，原`experiment.py`已逐字节保存为`_frozen_experiment_v1.py`，不作为公共运行入口。新的`experiment.py`仅限制允许的后端，沿用原HDBSCAN训练实现。核验比较历史调用中记录的原始哈希与冻结引擎文件；不修改历史哈希，也不改变已完成案例。BIRCH底层实现仅为历史数据审计保留。

缓存命中仍计入请求预算。整数参数可能导致多个编码坐标对应同一物理参数，分别记录编码参数、物理参数与不同分区；历史格式`actual_cuts_at_display`在本包表示实际后端拟合/提取数。HDBSCAN层次树成本另列。墙钟受缓存与并发影响，不作为冷启动公平效率对比。距离排序相同也不能保证EOM选簇相同，必须实测。

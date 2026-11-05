# 双曲空间实验：代码、图表与复现入口

本文件介绍在原始TPAMI AR-DBSCAN Fig.6欧氏复现基础上新增的工作。原始`README.md`、`reproduce_fig6.py`、`author_source/`和`reference/`保持原样；原README中的“不包含双曲迁移”描述原始复现包，后续扩展见本文件。

## 已完成的实验

| 算法 | 代码与运行说明 | 已完成案例 | 报告 | 双曲空间主图 |
|---|---|---:|---|---|
| DBSCAN | [hyperbolic/](hyperbolic/README.md) | 2640 | [报告](hyperbolic/report/报告.md) | [PNG](hyperbolic/report/fig6_H.png) / [PDF](hyperbolic/report/fig6_H.pdf) |
| AC凝聚聚类 | [ac_joint/](ac_joint/README.md) | 2800 | [报告](ac_joint/report/报告.md) | [PNG](ac_joint/report/fig6_AC_H.png) / [PDF](ac_joint/report/fig6_AC_H.pdf) |
| HDBSCAN | [density_cf/](density_cf/README.md) | 2800 | [报告](density_cf/report/报告.md) | [PNG](density_cf/report/HDBSCAN/fig6_HDBSCAN_H.png) / [PDF](density_cf/report/HDBSCAN/fig6_HDBSCAN_H.pdf) |

八个数据集为Aggregation、Compound、Pathbased、D31、Asymmetric、Skewed、Cure-t1、Cure-t2，使用seed0–9。完整矩阵包括欧氏/双曲主比较、固定欧氏区域及映射强度对照，各轮方法和预算详见对应报告。案例数指数据集×种子×空间×方法组合，不是独立聚类调用次数。

AC搜索单链接、全链接、平均链接、原生双曲Ward及距离阈值；HDBSCAN搜索`min_samples`和最小簇大小。均不搜索真实簇数K。主双曲几何为曲率−1的Poincaré球；双曲Ward使用Fréchet平方距离和的合并增量。

**BIRCH已按用户指示终止并淘汰。** 原有538例仅作归档，剩余2262例已取消，不参与当前结论。历史代码和报告保留以便追溯，公共实验入口拒绝BIRCH，默认调度及分析仅包含HDBSCAN。[终止记录](density_cf/report/retirement.json)。

## 安装依赖

命令在仓库根目录执行。已验证环境为Linux、Python3.10.22、PyTorch2.0 CPU。安装需要联网，PyG扩展在没有匹配wheel时需要C++编译工具。

```bash
uv venv --python 3.10.22 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'numpy==1.23.5' 'torch==2.0.0+cpu' 'setuptools==68.2.2' pip wheel
MAX_JOBS=2 .venv/bin/python -m pip install --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r hyperbolic/requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -r density_cf/requirements.txt
.venv/bin/python -m pip check
```

`hyperbolic/setup.sh`保留原云环境`/workspace/cluster`路径的安装流程；上面的命令可以在其他仓库路径执行。

## 从源码重建实验

逐例结果未包含在Git仓库中。完整训练需要较长运行时间和额外磁盘空间；可先使用各模块的`--datasets`、`--spaces`、`--seeds`参数缩小范围。

```bash
# 冻结样本、双曲几何和SE区域；不运行DBSCAN训练
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m ac_joint.bootstrap --workers 2

# 构造AC树库；同时提供当前HDBSCAN独立审计入口复用的区域元数据
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m ac_joint.prepare --workers 2
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m density_cf.experiment \
  --backends HDBSCAN --workers 4
.venv/bin/python -m density_cf.verify
.venv/bin/python -m density_cf.audit
.venv/bin/python -m density_cf.analyze
```

DBSCAN、AC完整执行与核验命令分别见[hyperbolic/README.md](hyperbolic/README.md)及[ac_joint/README.md](ac_joint/README.md)。报告中的旧训练引擎哈希保留历史来源，HDBSCAN公共入口的范围限制不改变已有训练规则。

仅执行HDBSCAN聚类时可以改用`density_cf.prepare`生成的轻量索引，见模块说明；当前独立后端审计入口复用`results/ac_joint/banks/`路径，以上完整流程包含该准备步骤。

## 上传范围与指标解释

仓库包含源码、依赖版本、说明、冻结报告、PNG/PDF图、汇总CSV与核验记录。`results/`中的完整候选、权重、距离矩阵、树库和逐例日志，以及`.venv/`和缓存均不上传。只读取仓库内的汇总表不能还原逐例模型，重新绘制需先生成相应案例结果。

Fig.6式全标签历史最大NMI用于离线展示；参数由20%支持标签奖励选择，补集标签不参与训练或选参。AC与HDBSCAN沿用完整训练、回顾性展示前30次训练请求，并记录全部训练与检测成本；DBSCAN图的缓存条目口径另见其报告，不能直接当作相同计算预算。

双曲迁移是否与欧氏效果相近，以各报告的预设配对检验结果为准，不因代码采用双曲距离就宣称效果相同或更好。

# TPAMI AR-DBSCAN Fig.6 复现

本仓库包含此前 Fig.6 复现所用代码、8个数据集及CPU环境配置。

## 文件

- `reproduce_fig6.py`：训练、记录候选和绘图入口；只把原脚本的源码路径改为包内相对路径。
- `author_source/`：作者源码（包括自带silearn）及`data/`中的8个二维数据集。
- `requirements-cpu.txt`、`environment.yml`：Python3.10、PyTorch2.0 CPU环境。
- `reference/summary.csv`、`reference/fig6_with_reconstructed_baselines.png`：原640例的汇总与结果图。
- `reference/protocol.json`：原复现参数、种子、作者源码SHA256与口径说明。
- `reference/author_local_changes.patch`：原作者代码的本地兼容性修改记录。
- `verify_bundle.py`、`SHA256SUMS.txt`：文件完整性校验。

## 安装与运行

已验证环境是Linux CPU；Windows可使用WSL2。依赖安装需要联网下载，不包含离线安装包。

```bash
conda env create -f environment.yml
conda activate tpami-fig6
python verify_bundle.py
python -u reproduce_fig6.py --output results/full_run --workers 8
```

内存不足时将`--workers`设为2或1。完整任务为8数据集×10种子×8方法=640例。

单案例检查仍采用完整默认训练，但只执行Pathbased、AR-DBSCAN、seed0：

```bash
python -u reproduce_fig6.py --output results/smoke --datasets Pathbased --methods AR-DBSCAN --seeds 0 --workers 1
```

输出目录包含模型、候选、输入掩码、预测和Fig.6图；`results/`已在`.gitignore`中忽略。

## 数据集与方法

数据集：Aggregation、Compound、Pathbased、D31、Asymmetric、Skewed、Cure-t1、Cure-t2。

方法：AR-DBSCAN、DRL-DBSCAN（同源码单区域子模型）、Rand、BO-TPE、Anneal、PSO、GA、DE。

六个搜索基线是论文设置的重建版本；原作者未提供完整基线实验脚本，BDE缺失，不包含该曲线。

作者默认3层、episode_num=15（实际range(1,15)，最多14局）、step_num=30、batch=16、20%奖励标签。

横轴从1开始，计每智能体首次出现的新训练缓存条目；detect和缓存命中不推进此轴。局部支持标签NMI选择incumbent，合并后离线计算全标签NMI并取历史最大。参数、训练和原代码已知mask/标签合并行为均保留，以对应原复现。

本包用于复现已交付的欧氏Fig.6实验；不包含双曲迁移或后来机制审计修改，也不声称逐点精确复现论文原图。

## 源码来源

作者仓库：[RingBDStack/AR-DBSCAN](https://github.com/RingBDStack/AR-DBSCAN)

固定commit：`b9cfcd41fc1d34348e75352ec457dc1d08df8e02`。

作者35个Python文件与原复现快照逐字节一致。本包保留作者README和兼容性补丁供溯源。

## GitHub上传

解压后上传本目录中的源码、数据和说明；原640例逐例日志、权重、结果JSON与重复源码快照已移出此精简包。

`reference/`仅含汇总，不能直接使用`--plot-only`重画；需要先运行完整训练生成逐例结果。

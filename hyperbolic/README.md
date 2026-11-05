# 欧氏—双曲 Fig.6 受控实验

原 `reproduce_fig6.py`、`author_source/`、数据和历史汇总均保留。本目录新增统一审计的欧氏/双曲实验，不把旧缺陷修复误称为空间效应。

本次2640例实验已完成，独立产物核验通过。[完整中文报告](report/报告.md)、[欧氏图](report/fig6_E.pdf)、[双曲图](report/fig6_H.pdf)与汇总数据保存在 `report/`。大体积逐例候选、权重和距离矩阵位于忽略目录 `results/hyperbolic/`。报告副本为本次冻结结果，后续新运行的分析输出写入对应输出目录，不自动覆盖该副本。

## 环境和执行

工作目录 `/workspace/cluster`。使用已安装的 `.venv/bin/python`（Python 3.10、PyTorch 2.0 CPU）。每个云任务已有隔离环境，直接使用本 checkout，不另建 Git worktree。

```bash
.venv/bin/python -m hyperbolic.prepare --output results/hyperbolic
.venv/bin/python -m unittest hyperbolic.test_geometry -v
.venv/bin/python -u -m hyperbolic.experiment --output results/hyperbolic \
  --datasets Compound D31 --seeds 0 --spaces E H --workers 2
.venv/bin/python -u -m hyperbolic.experiment --output results/hyperbolic --workers 4
.venv/bin/python -u -m hyperbolic.replay --output results/hyperbolic
.venv/bin/python -m hyperbolic.verify --output results/hyperbolic --workers 4
.venv/bin/python -m hyperbolic.analyze --output results/hyperbolic
```

正式矩阵 2640 例：E/H 各880，固定欧氏区域240，两档敏感性合计640。每一例是一个方法×数据×种子×空间，不是单次DBSCAN。成功案例可续跑跳过；失败案例保留且阻止静默覆盖，修正实现后使用新输出根目录。试运行只有在最终训练源码、输入和配置一致时才能计入正式结果。

已有训练运行时，可另启 `.venv/bin/python -u -m hyperbolic.finish --output results/hyperbolic`，等待完整E候选后回放，并在2640例齐备后执行独立产物核验及分析。该监视器不会替代训练进程，也不能在云任务快照中保活。

## 固定协议

二维映射明确为 `z=x-mean(x)`、`R=max||z||`、`a=atanh(r_max)/R`、`p=tanh(a||z||)z/||z||`（零向量映到零）。曲率−1下使用 `d(p,q)=2 asinh(||p-q||/sqrt((1-||p||²)(1-||q||²)))`。全局阈值尺度 `s_H=Q90(D_H)/Q90(D_E)`，搜索坐标 `u=eps_H/s_H`。这些定义锁定指数映射的系数约定。

- 八数据集、seed0–9；按原Python `random.sample`语义重建样本顺序和20%支持ID。已与原加载器核对；未提供历史 `input.npz`，不声称逐字节恢复历史输入。
- 归一化坐标、原始样本ID、支持ID写入 `inputs/`；每个数据、每档映射只计算一次全局中心、尺度和距离，种子只改变顺序。
- E为精确float64欧氏距离；H为曲率−1、二维径向Poincaré映射。主r_max=.5，敏感性.2/.8。全数据共享Q90距离比例，不逐区域重定标。
- 统一稳定排序的精确kNN；作者SE优化器、k扫描、边权和不确定性区域分配保留。小模块最近正常点归并采用相应测地距离。无正常模块时显式失败。
- 模块ID一次映射、最终非噪声簇全局唯一编号、噪声−1。小于2个支持点时奖励固定0；不补抽标签。
- 训练器仅持有允许支持标签。完整标签、支持补集标签均在训练外用于评价，不参与停止、递归、选择或映射强度。
- 原DrlDbscan训练/检测和TD3代码通过运行时适配器复用；三层预初始化、range(1,15)、step30、batch16、探索std2和原早停保留。不进行辅助真K的KMeans。
- 位置特征为 `center_E + log_0(q)/a`，距离状态除以全局s_H。Fréchet均值使用Karcher迭代和回溯，残差 `||mean(log_q(p_i))||_q` 阈值1e−7（对应半平方距离目标的梯度范数），最多512次；失败不回退。
- 极小区域的整数步长下限设为1，避免原整数步长为0；记录为统一审计差异。回放缓冲容量4096，固定协议每层至多406条，断言不发生覆盖，采样机制不变。
- MA随机组使用同一初始(.5,ceil(n/8))；随后29次在以初始中心为锚点的末层物理网格独立均匀采样，不使用递归窗口。仓库未包含旧机制审计源码，因此不宣称复用了未提供的规则。
- 冻结组只阻止Actor optimizer.step；保留Critic、探索、回放和目标网络代码，并逐层验证Actor参数不变。
- 显示轴为前30个新训练缓存条目/区域，不足持有最后结果。每区域支持奖励选择incumbent，再合并、离线评价；先合并后取全标签历史最大值。
- AUC30沿用原复现的平均曲线值定义 `mean(H1,...,H30)`，不使用另一种梯形积分归一化口径。
- 拟合日志记录所有实际训练/检测DBSCAN调用；缓存命中不伪称唯一拟合。共享分区成本单列。绘图另给实际拟合数横轴。
- 案例CPU/墙钟包含训练、保存产物和离线评价，不冒称纯训练时间；逐区域fit_seconds单独记录DBSCAN原生拟合耗时。进程RSS为累计峰值，不能作为精确逐案例峰值。
- 新图阴影使用标准差并标明；原图入口均值±方差口径保留。

## 三维实现校验与浮点边界

同高度上半空间的解析控制满足 `d_H=2 asinh(a*d_E/2)`。从实际三维球坐标独立计算距离，再运行DBSCAN。每个保存的E候选均可通过replay重新拟合。

float64坐标转换在恰好等于eps的边上可能产生1ULP差异。原始差异、点ID、距离间隙和原生DBSCAN核心点/噪声/ARI均保存；不放宽eps。几何探针对可疑边另用有理数精确重算原坐标和解析球映射，独立验证两边的闭邻域判定。该重算仅用于三维控制，不修改二维主实验的原生float64距离。

## 主要统计判据

主指标：第30显示步的支持奖励incumbent，在支持补集上的NMI。主H与E的MA-TD3八数据等权宏平均配对差，90%分层种子bootstrap区间须完全落在[−.02,.02]，且至少7/8数据均值差绝对值≤.05。10000次重采样，只描述这八个固定基准的运行随机性。未齐80对时不判断等效。

补充支持补集ARI、噪声率、簇数、全标签历史最大曲线、曲线平均面积、E/H最终预测ARI、分区层级质量、真实拟合数和Actor/Critic更新数。

## 输出

- `inputs/`：冻结输入和原始样本ID。
- `geometry/`：点坐标、距离矩阵、映射和尺度。
- `validation/`：几何探针、边界差异、每候选严格回放。
- `partitions/`：原始编码树、小模块归并、agent区域及选k记录。
- `cases/`：逐区域候选、支持ID、拟合成本、网络权重、全局预测和结果。
- `invocations/`：每次调用清单、训练源码哈希和完整案例预算。
- `analysis/`：汇总CSV、配对差异、自助区间、图和中文报告。

检查点文件只加载本实验生成且可信的内容；不会从网络加载模型。

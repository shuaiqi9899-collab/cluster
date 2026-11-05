# 80条目录的双曲空间 Fig.6 实验

本轮仅运行双曲DBSCAN、AC、HDBSCAN，每个纳入数据集跑seed0–9。不生成欧氏距离矩阵、欧氏聚类、固定欧氏区域或E/H对照；BIRCH算法不恢复。原`hyperbolic/`、`ac_joint/`、`density_cf/`的冻结代码与报告保持不变。

## 当前数据范围

`catalog.xlsx`为用户提供的原始80条目录，`catalog.json`为只读提取的数据字段；附件中的研究建议不作为执行命令。最新下载/版本状态在`results/benchmarks80/datasets.json`及`analysis/报告.md`。

初次清点结果：47项已取得并验证匹配数据；13项原始N超过10000暂缓；4项目录已注明缺少精确PRIME包；4项精确特征包尚未核实；12项官方下载被当前网络策略阻断。网络阻断不等于数据不存在。没有将不匹配的小版本或抽样样本冒充全量数据。

取得的数据包括UCI小型表格、SIPU形状、FCPS二维/三维结构、作者形状数据、3Sources三视图、Cora/CiteSeer节点属性。所有数据记录下载地址、来源版本、文件SHA256、N/d及标签可用性。Cure-t2原始作者文件保留4000正常点加200噪声点；CiteSeer保留3327节点，其中没有参考标签的节点参与聚类但不参与奖励和外部评价。图的原始边文件也保存，主实验不使用边。

## 固定协议

- 数据不抽样。完整数据N>10000先暂缓；其余仍需通过实际内存与运行成本检查。
- 形状数据保留原维度并逐坐标MinMax。其他稠密数据中位数填补、标准化；多视图逐视图标准化并按`1/sqrt(d_view*V)`等能量拼接。高维表示使用无标签PCA/SVD，最高64维。随机化变换种子固定20261105，不使用类别标签。
- 径向Poincaré映射，曲率−1、最大点半径0.5。唯一距离矩阵是原生双曲距离，存为float64。尺度为固定无标签点对样本的双曲Q90，不依赖欧氏距离比例。
- 每个种子按`random.Random(seed)`先抽完整点序，再在有标签ID中均匀抽20%作为支持；不按类别重抽。后端统一采用原始ID规范顺序处理距离并列，种子改变支持掩码和网络随机数。
- SE/k扫描沿用原作者实现，将kNN边长替换为原生双曲距离；不确定性分组采用标量相邻连通规则。十个SE分区在三个后端间共享。
- AC联合搜索single/complete/average/native Fréchet-SSE Ward与阈值。Ward绝不回退到切空间或欧氏递推。
- HDBSCAN搜索min_samples四类1/5/10/20（局部截断）与对数min_cluster_size，固定EOM、alpha1、epsilon0、不选单根簇。
- DBSCAN搜索两个数值坐标：eps在全局最小正H距离一半到H-Q90之间对数映射，min_samples在1到ceil(n_local/4)之间线性映射后取整。
- Actor/Critic沿用原隐藏层与TD3设置。DBSCAN全局7维、局部4维、5动作；AC/HDBSCAN全局8维、局部4维、7动作。局部位置的二维投影只作网络状态，所有聚类距离、SE和Fréchet目标保留完整表示维度。
- 完整三层独立网络训练，最多14局/层、每局30步、batch16，原探索std2及早停规则保留。回顾性展示每区域前30次训练请求；探针、重复请求、缓存命中均计入。训练/检测总请求、实际拟合、唯一参数、网络更新和共享准备成本另记。
- 三个后端各有4个计入预算的初始探针。DBSCAN探针为归一化参数坐标(.5,.5)、(.25,.25)、(.75,.25)、(.25,.75)；AC/HDBSCAN为四类u=.5、按seed轮换。
- DBSCAN包含11种搜索方法，AC/HDBSCAN各12种，见`protocol.json`。每数据集目标为350例；初始47项候选目标16450例。35例单种子预跑单独保存在`pilot/`，不计入正式完成数。
- 训练只取得支持标签；补集NMI/ARI只离线评价。全标签历史最大NMI为Fig.6式展示，另输出支持奖励incumbent的非单调补集曲线与完整训练结果。

## 资源与完成定义

当前4CPU、32GiB环境默认3个数据集工作进程，各1个BLAS线程。每数据集先准备十个分区与全部四种原生AC树，再开始三个后端的训练。准备超过1800秒、单个完整训练案例超过900秒或单进程RSS超过8GiB，将整个数据集记为`deferred_resource`，保留已有文件，不缩短训练、改用抽样或替换算法。磁盘保留5GiB余量。

`benchmarks80.finish`会等待首轮队列退出，再将仅因8GiB并发内存配额暂缓的条目以单进程、24GiB预算复核一次，最后导出本地报告快照。运行时长预算保持不变；并发配额超限不直接等同于32GiB云端无法运行。所有阶段与中断尝试保留记录，不自动Git发布。

`completed`要求三个后端所有十种子、350例全部成功，并完成独立产物核验。资源暂缓、网络阻断、数据包未核实和代码/数值失败分开记录；不是完成。正式图表与三后端比较只纳入已完成并核验的整数据集；中间CSV另外标明种子数及执行状态。

每次实际训练请求、候选预测和最终参数均保存；逐例指标会从支持奖励和预测文件独立重算。策略权重用gzip无损压缩，保留SHA256。原生H距离另以独立arccosh公式抽查。单种子功能预跑已核对35例、3128个支持奖励；另有7项新流程测试覆盖三维/60维/64维、标准后端一致性、预算和标签隔离。

## 执行与恢复

沿用现有依赖：`hyperbolic/requirements-lock.txt`和`density_cf/requirements.txt`，无新服务或GPU要求。

```bash
# 下载需要网络。UCI/SIPU等官方域名仍需环境策略允许；已允许的GitHub镜像可先使用。
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks80.acquire --workers 4

# 数据准备与后端门禁检查；先需准备代表性数据（参见test_pipeline.py）
.venv/bin/python -m benchmarks80.prepare 01-iris
.venv/bin/python -m benchmarks80.prepare 42-hepta
.venv/bin/python -m benchmarks80.prepare 09-sonar
.venv/bin/python -m benchmarks80.prepare 62-3sources-complete-3v
.venv/bin/python -m unittest benchmarks80.test_pipeline -v

# 查看任务范围，不启动训练
.venv/bin/python -m benchmarks80.run --dry-run

# 运行所有当前已就绪数据，自动跳过已完成数据，冻结源码并逐数据集核验
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m benchmarks80.run --workers 3

# 可在另一进程等待队列、串行复核内存配额暂缓项并导出快照
.venv/bin/python -m benchmarks80.finish

# 真实进程状态与实际案例计数（不能仅相信快照中的旧PID）
.venv/bin/python -m benchmarks80.status

# 重画已完成数据的图与汇总，不做任何聚类拟合
.venv/bin/python -m benchmarks80.analyze

# 导出可提交到Git的中途/最终本地快照；不会自动Git提交或推送
.venv/bin/python -m benchmarks80.snapshot
```

数据、距离矩阵、树、候选和权重位于忽略目录`results/benchmarks80/`；轻量源码、目录原表和说明位于本包。`protocol.json`冻结训练源码哈希，恢复前验证。若改变训练/表示语义，保留原结果并使用新版本输出目录，不能混用旧案例。发布/切换云环境不会保留进程；任务状态与成功案例可以恢复，续跑前必须核实进程、源码和资源，避免重复启动。

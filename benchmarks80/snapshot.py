"""Export an explicitly partial local report snapshot; never publishes remotely."""
import argparse,csv,fcntl,hashlib,json,shutil,time
from pathlib import Path
from .common import OUT,ROOT,write
from .analyze import analyze


def export(out=OUT,destination=ROOT/'benchmarks80/report'):
 out=Path(out);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 lock=(out/'snapshot_export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 preview=out/'snapshot';preview.mkdir(exist_ok=True);shutil.copy2(out/'datasets.json',preview/'datasets.json')
 for name in ['cases','execution']:
  p=preview/name
  if not p.exists():p.symlink_to((out/name).resolve(),target_is_directory=True)
 analyze(preview,plots=True);shutil.copytree(preview/'analysis',destination,dirs_exist_ok=True)
 for name in ['inventory.json','inventory.csv','protocol.json','data_label_audit.json']:
  if (out/name).exists():shutil.copy2(out/name,destination/name)
 if (ROOT/'results/benchmarks80_tests.log').exists():shutil.copy2(ROOT/'results/benchmarks80_tests.log',destination/'pipeline_tests.txt')
 pilot=out/'pilot/validation/01-iris.json'
 if pilot.exists():shutil.copy2(pilot,destination/'pilot_validation.json')
 comparison=destination/'backend_comparisons.csv'
 if comparison.exists():
  with comparison.open() as f:completed=len(list(csv.DictReader(f)))
 else:completed=0
 status=json.loads((destination/'status.json').read_text())
 (destination/'README.md').write_text(f'''# 运行中快照

本文件是H-only批量任务的导出快照，**不代表所有任务已经完成**。

- 输入已核对{status['ready_datasets']}条，候选目标{status['candidate_cases']}例；每数据集三个后端各10种子。
- 本快照读取{status['passed'] if 'passed' in status else status['successful_cases']}个成功案例，其中{completed}个数据集已完成并核验完整十种子，共{completed*350}例。
- 精确包未核实、规模超限和网络阻断分别列明；主图和三后端比较仅使用本次快照中350例齐备且核验完成的数据集。
- 数据队列可能仍在继续；GitHub上的导出文件不会实时更新，也不能证明云进程仍存活。

[目录清单](inventory.csv) · [详细原因与来源](inventory.json) · [快照报告及图](报告.md) · [三后端配对均值比较](backend_comparisons.csv)

实时状态请执行`.venv/bin/python -m benchmarks80.status`。原始数据、逐例候选、权重和距离矩阵不上传到GitHub。
''')
 source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'benchmarks80').glob('*') if p.is_file()}
 artifacts={str(p.relative_to(destination)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in destination.rglob('*') if p.is_file() and p.name!='source_manifest.json'}
 write(destination/'source_manifest.json',dict(training_state='in_progress_snapshot',snapshot_unix=time.time(),verified_datasets=completed,source_sha256=source,report_files=artifacts))
 lock.close();print('Snapshot exported:',completed,'complete verified datasets',flush=True)
 return completed


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--destination',type=Path,default=ROOT/'benchmarks80/report');a=p.parse_args();export(a.output,a.destination)

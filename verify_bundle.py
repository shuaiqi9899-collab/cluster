"""Verify all packaged file hashes; uses only Python's standard library."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent
lines = (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
for line in lines:
    expected, name = line.split('  ', 1)
    path = root / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('FAILED: ' + name)
print('Verified', len(lines), 'files.')

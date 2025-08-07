import os
import subprocess
import sys
import time
from pathlib import Path

_cwd = Path(os.getcwd())
_result = subprocess.run(["git", "diff-index", "--quiet", "HEAD",  "--"]).returncode
_clear = True

if _result == 128:
    _cwd = Path(__file__).parent
    _result = subprocess.run(["git", "diff-index", "--quiet", "HEAD",  "--"], cwd=_cwd).returncode
    _clear = False



_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=_cwd).decode('utf-8').strip()
_test_no_fail=False


for l in _status.splitlines():
    v = l.split()
    if v[0] == '??':
        _clear = False

_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_cwd).decode('utf-8').strip()

dirname = str(int(time.time())) + "-" + _hash
shorthash = _hash[0:8]

# Don't error out inside pytest
if _result != 0 or not _clear or _test_no_fail:

    if os.environ.get('OVERRIDE_UNCLEAN_REPO') is None:
        if _result != 0:
            print("Error, uncommitted changes")
        if not _clear:
            print("Error, untracked files")

        print("override with")
        print("OVERRIDE_UNCLEAN_REPO=1 \x1B[1mcommand\x1B[0m")
        sys.exit(1)

    print()
    print()
    print("\x1B[31;1m****************************\x1B[0m")
    print("\x1B[31;1mWarning, uncommitted changes\x1B[0m")
    print("\x1B[31;1m****************************\x1B[0m")
    print()
    print()
    dirname += "-unclean"
    shorthash += "-unclean"

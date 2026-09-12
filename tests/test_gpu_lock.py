import subprocess
import sys

from gsdb.gpu_lock import gpu_session


def test_gpu_lock_blocks_other_process_and_releases_on_exit(tmp_path):
    path=tmp_path/'gpu.lock'
    code='''from pathlib import Path
import sys
from gsdb.gpu_lock import gpu_session
print('starting',flush=True)
with gpu_session(Path(sys.argv[1])):
    print('acquired',flush=True)
'''
    with gpu_session(path):
        with gpu_session(path):
            child=subprocess.Popen([sys.executable,'-c',code,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            assert child.stdout.readline().strip()=='starting'
            assert child.stdout.readline().startswith('Waiting')
            assert child.poll() is None
    output,error=child.communicate(timeout=15)
    assert child.returncode==0,error
    assert 'acquired' in output

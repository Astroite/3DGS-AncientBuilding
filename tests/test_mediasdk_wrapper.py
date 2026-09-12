"""Exercise the retained hardware wrapper with a stub CLI, never a real SDK."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import cv2
import numpy as np
import pytest

from gsdb.media import sha256_file


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell hardware wrapper')
@pytest.mark.parametrize('candidate_fps,success', [(5, True), (2, False)])
def test_hardware_wrapper_uses_current_sampling_and_rejects_wrong_lineage(tmp_path, candidate_fps, success):
    powershell = shutil.which('pwsh')
    if not powershell:
        pytest.skip('PowerShell 7 unavailable')
    app = tmp_path / 'APP'
    scripts = app / 'scripts'
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'test-mediasdk-helper.ps1'
    shutil.copyfile(source, scripts / source.name)
    (scripts / 'session.ps1').write_text("# Test seam: environment variables are supplied by the test.\n")
    (app / 'gsdb.ps1').write_text("""
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$GsdbArgs)
ConvertTo-Json -InputObject @($GsdbArgs) -Compress | Add-Content -LiteralPath $env:TEST_CALLS
if ($GsdbArgs[0] -eq 'media') { Write-Output '{}'; exit 0 }
if ($GsdbArgs[0] -eq 'preprocess') { Write-Output 'RUN_ID=run-001'; exit 0 }
exit 9
""")
    helper = tmp_path / 'fixture-helper.exe'
    pe = bytearray(128)
    struct.pack_into('<H', pe, 0, 0x5A4D)
    struct.pack_into('<I', pe, 0x3C, 64)
    struct.pack_into('<IH', pe, 64, 0x00004550, 0x8664)
    helper.write_bytes(pe)  # Read as a header only; never executed.
    data = tmp_path / 'Data'
    scene = data / 'location' / 'scene'
    run = scene / 'run-001'
    run.mkdir(parents=True)
    (run / 'manifest.yaml').write_text('prepared_relative_path: prepared/capture/digest\n')
    dataset = scene / 'prepared' / 'capture' / 'digest'
    dataset.mkdir(parents=True)
    frames = []
    for i in range(2):
        name = f'frame_{i:06d}.jpg'
        cv2.imwrite(str(dataset / name), np.full((8, 16, 3), 127, np.uint8))
        frames.append(dict(file=name, sha256=sha256_file(dataset / name), width=16, height=8, source_frame_index=i*6))
    meta = dict(schema_version=2, integrity='complete', frames=frames, preparation_hash='digest', dataset_sha256='hash',
                lineage=dict(capture_id='capture', source_kind='insta360_insv', source_files=[{}],
                             sampling=dict(mode='fixed_rate_v1', candidate_fps=candidate_fps), candidate_frame_count=2,
                             helper_version='0.2.0', sdk_version='3.1.5'))
    (dataset / 'dataset.json').write_text(json.dumps(meta))
    calls = tmp_path / 'calls.jsonl'
    env = os.environ.copy()
    env.update(GSDB_DATA_ROOT=str(data), GSDB_MEDIA_HELPER=str(helper),
               INSTA360_MEDIA_SDK_ROOT=str(tmp_path), TEST_CALLS=str(calls))
    result = subprocess.run([powershell, '-NoProfile', '-File', str(scripts / source.name),
        '-LocationId', 'location', '-SceneId', 'scene', '-CaptureId', 'capture', '-RunId', 'run-001'],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding='utf-8')
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    operations = [json.loads(line) for line in calls.read_text(encoding='utf-8-sig').splitlines()]
    assert operations == [
        ['media', 'probe', 'location', 'scene', 'capture'],
        ['preprocess', 'location', 'scene', 'capture', '--candidate-fps', '5', '--run-id', 'run-001', '--resume']]

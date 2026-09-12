import json
import sys
from pathlib import Path

import pytest

from gsdb.training import train_package


def test_postshot_dry_run_can_execute_and_zero_exit_without_model_fails(tmp_path,monkeypatch):
    from gsdb import training,postshot
    package=tmp_path/'package'
    package.mkdir()
    (package/'dataset.json').write_text('{}')
    monkeypatch.setenv('GSDB_GSPLAT_PYTHON',sys.executable)
    monkeypatch.setattr(training,'validate_package',lambda *a:dict(images=[dict(split='train')]))
    monkeypatch.setattr(postshot,'postshot_executable',lambda:Path('postshot-cli.exe'))
    monkeypatch.setattr(postshot,'_windows_argument',str)
    def adapter(source,output):
        output.mkdir()
        (output/'adapter.json').write_text(json.dumps({'package_sha256':training.sha256_file(source/'dataset.json')}))
    monkeypatch.setattr(training,'postshot_adapter',adapter)
    monkeypatch.setattr(training,'validate_postshot_adapter',lambda *a:None)
    def execute(command,log,*a,**k):
        assert command[command.index('--profile')+1]=='Splat ADC'
        assert '--no-recenter-points' in command
        assert command[command.index('--max-sh-degree')+1]=='3'
        log.write_text('Postshot Studio license required')
        return dict(elapsed_seconds=0.)
    monkeypatch.setattr(training,'run_logged',execute)
    output=tmp_path/'experiment'
    prepared=train_package(package,output,'postshot',dry_run=True)
    assert prepared['status']=='prepared'
    with pytest.raises(RuntimeError,match='license/log'):
        train_package(package,output,'postshot')
    saved=json.loads((output/'dispatch.json').read_text())
    assert saved['status']=='failed'
    assert (output/'postshot-input'/'adapter.json').is_file()
    with pytest.raises(FileExistsError):
        train_package(package,output,'postshot')


def test_zero_training_steps_is_rejected(tmp_path,monkeypatch):
    from gsdb import training
    monkeypatch.setattr(training,'validate_package',lambda *a:dict(images=[dict(split='train')]))
    with pytest.raises(ValueError,match='positive'):
        train_package(tmp_path/'package',tmp_path/'out',steps=0)

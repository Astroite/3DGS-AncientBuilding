"""Reentrant, process-wide per-Run exclusion for stages and retention."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import threading

_mutex = threading.RLock()
_local = threading.local()


@contextmanager
def run_session(work: Path):
    from .retention import safe_path
    with _mutex:
        key = str(work.absolute())
        active = getattr(_local, 'active', set())
        if key in active:
            yield
            return
        lock = safe_path(work, '.run.lock')
        with lock.open('a+b') as stream:
            if lock.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as error:
                raise RuntimeError('Run is busy; wait for its active stage before cleanup/resume') from error
            _local.active = active | {key}
            try:
                yield
            finally:
                _local.active = active
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def run_locked(function):
    @wraps(function)
    def wrapped(scene_path, run, *args, **kwargs):
        if getattr(run.config, 'schema_version', 0) != 6:
            return function(scene_path, run, *args, **kwargs)
        with run_session(scene_path / run.id):
            # A caller may have loaded before acquiring the lock.
            from .runs import load_run
            run = load_run(scene_path, run.id)
            return function(scene_path, run, *args, **kwargs)
    return wrapped


def run_cli_locked(function):
    @wraps(function)
    def wrapped(location_id, scene_id, run_id, *args, **kwargs):
        from .cli import _scene
        from .runs import load_run
        scene = _scene(location_id, scene_id)
        run = load_run(scene, run_id)
        if run.config.schema_version == 6:
            with run_session(scene / run_id):
                return function(location_id, scene_id, run_id, *args, **kwargs)
        return function(location_id, scene_id, run_id, *args, **kwargs)
    return wrapped


def run_dir_locked(function):
    @wraps(function)
    def wrapped(run_dir, *args, **kwargs):
        load_run = function.__globals__['load_run']
        run_dir = Path(run_dir).absolute()
        run = load_run(run_dir.parent, run_dir.name)
        if run.config.schema_version == 6:
            with run_session(run_dir):
                return function(run_dir, *args, **kwargs)
        return function(run_dir, *args, **kwargs)
    return wrapped

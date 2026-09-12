import pytest


@pytest.fixture(autouse=True)
def isolated_unit_test_gpu_lock(tmp_path,monkeypatch):
    """Unit tests mock GPU commands; keep their real locks off production jobs."""
    from gsdb import gpu_lock
    monkeypatch.setattr(gpu_lock,'GPU_LOCK_PATH',tmp_path/'unit-test-gpu.lock')

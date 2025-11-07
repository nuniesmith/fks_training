from shared_python import list_apps  # type: ignore

def test_worker_registered():
    assert "worker" in list_apps()

def test_register_duplicate_raises():
    import pytest
    from shared_python.runtime.registry import register_app as _reg
    from shared_python.runtime.registry import _REGISTRY  # type: ignore
    # Ensure duplicate triggers error
    with pytest.raises(ValueError):
        _reg("worker", lambda: None)
    # registry unchanged
    assert "worker" in _REGISTRY

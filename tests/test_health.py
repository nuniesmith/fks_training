def test_dummy_import():
    # Basic import & attribute presence test (doesn't start server)
    import importlib
    m = importlib.import_module('main')
    assert hasattr(m, 'main')

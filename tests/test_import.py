def test_import_training():
    import importlib
    mod = importlib.import_module("fks_training.main")
    assert mod is not None

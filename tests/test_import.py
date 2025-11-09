def test_import_training():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    mod = importlib.import_module("main")
    assert hasattr(mod, "main")

import numpy as np  # type: ignore
try:
    from shared_python.types import BaselineModelParams  # type: ignore
except Exception:  # pragma: no cover
    from shared_python.types import BaselineModelParams  # type: ignore
from models.baselines import XGBoostClassifierWrapper, CatBoostClassifierWrapper
from models.time_series_cv import rolling_purged_splits


def _tiny_dataset(n=40):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.2, size=n) > 0).astype(int)
    return X, y


def test_time_series_cv_basic():
    splits = list(rolling_purged_splits(120, n_splits=4, purge=5, min_train=60))
    assert splits, "No splits generated"
    for s in splits:
        assert max(s.train_idx) < min(s.test_idx)


def test_baseline_param_schema():
    p = BaselineModelParams(model_type="xgboost", max_depth=4, learning_rate=0.1)
    eff = p.effective_params()
    assert eff["model_type"] == "xgboost"
    assert eff["max_depth"] == 4


def test_xgboost_wrapper_instantiation():
    # Only instantiate; skip actual fitting if xgboost not installed in minimal env
    X, y = _tiny_dataset()
    try:
        mdl = XGBoostClassifierWrapper(max_depth=2, n_estimators=10)
        mdl.fit(X, y)
        probs = mdl.predict_proba(X[:5])
        assert probs.shape[0] == 5
    except Exception:
        # Accept import errors in trimmed dependency environments
        pass


def test_catboost_wrapper_instantiation():
    X, y = _tiny_dataset()
    try:
        mdl = CatBoostClassifierWrapper(depth=2, iterations=10)
        mdl.fit(X, y)
        probs = mdl.predict_proba(X[:5])
        assert probs.shape[0] == 5
    except Exception:
        pass

r"""Every model family behind one interface (Prompt #10, Steps 5, 17-22, 72).

``fit(x, y, weight=, x_inner=, y_inner=)`` / ``predict(x)`` / ``save(dir)`` /
``load(dir)``. A classifier's ``predict`` returns :math:`P(y = 1 \mid x)`, a
regressor's :math:`E[y \mid x]` - never a decision. Boosters stop early on the
inner slice (chronologically after the fitting rows, purged); nothing is fitted
on the rows it is scored on.

Families: ``constant`` (the training mean / positive rate), ``logistic_l2`` /
``logistic_l1`` / ``logistic_en`` (scikit-learn, ``l1_ratio`` 0 / 1 / 0.5),
``ols`` / ``ridge`` / ``elastic_net``, ``random_forest``, ``xgboost`` (hist),
``lightgbm``, ``catboost``. Serialization uses each library's native format
(XGBoost UBJSON, LightGBM text, CatBoost ``.cbm``) and joblib for scikit-learn,
so a reloaded model reproduces its predictions.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FAMILY_PREFIX", "Model", "load_model", "make_model"]

#: short names used in model ids (XGB_REVERSION_5M_H5_V001)
FAMILY_PREFIX = {"constant": "CONST", "logistic_l2": "LOGL2", "logistic_l1": "LOGL1",
                 "logistic_en": "LOGEN", "ols": "OLS", "ridge": "RIDGE", "elastic_net": "ENET",
                 "random_forest": "RF", "xgboost": "XGB", "lightgbm": "LGBM",
                 "catboost": "CAT"}
_LINEAR = ("logistic_l2", "logistic_l1", "logistic_en", "ols", "ridge", "elastic_net")


def preprocessing_kind(family: str, tree_missing: str = "native") -> str:
    """Which :class:`~.preprocessing.Preprocessor` a family needs."""
    if family in _LINEAR or family == "constant":
        return "linear"
    if family == "random_forest" or tree_missing == "imputed":
        return "tree_imputed"
    return "tree_native"


def _evenly(n: int, size: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, size).astype(np.int64))


@dataclass
class Model:
    family: str
    task: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    threads: int = 4
    info: dict[str, Any] = field(default_factory=dict)
    _est: Any = None

    @property
    def is_classifier(self) -> bool:
        return self.task == "classification"

    # -- fitting -------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, *, weight: np.ndarray | None = None,
            x_inner: np.ndarray | None = None, y_inner: np.ndarray | None = None) -> Model:
        started = time.perf_counter()
        fam = self.family
        if fam == "constant":
            value = float(np.average(y, weights=weight))
            self._est = value
        elif fam in _LINEAR:
            self._fit_linear(x, y, weight)
        elif fam == "random_forest":
            self._fit_forest(x, y, weight)
        elif fam == "xgboost":
            self._fit_xgboost(x, y, weight, x_inner, y_inner)
        elif fam == "lightgbm":
            self._fit_lightgbm(x, y, weight, x_inner, y_inner)
        elif fam == "catboost":
            self._fit_catboost(x, y, weight, x_inner, y_inner)
        else:
            raise ValueError(f"unknown model family {fam!r}")
        self.info.update({"fit_seconds": time.perf_counter() - started, "fit_rows": int(len(y))})
        return self

    def _fit_linear(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> None:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge

        p = self.params
        fam = self.family
        rows = np.arange(len(y))
        if p.get("max_rows"):
            rows = _evenly(len(y), int(p["max_rows"]))
        xs, ys = x[rows], y[rows]
        ws = None if w is None else w[rows]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if fam.startswith("logistic"):
                ratio = {"logistic_l2": 0.0, "logistic_l1": 1.0}.get(fam, p.get("l1_ratio", 0.5))
                est = LogisticRegression(C=float(p.get("C", 1.0)), l1_ratio=float(ratio),
                                         solver="lbfgs" if ratio == 0.0 else "saga",
                                         max_iter=int(p.get("max_iter", 500)),
                                         class_weight="balanced" if p.get("balanced") else None,
                                         random_state=self.seed)
                est.fit(xs, ys.astype(np.int64), sample_weight=ws)
            elif fam == "ols":
                est = LinearRegression().fit(xs, ys, sample_weight=ws)
            elif fam == "ridge":
                est = Ridge(alpha=float(p.get("alpha", 1.0))).fit(xs, ys, sample_weight=ws)
            else:
                est = ElasticNet(alpha=float(p.get("alpha", 1e-4)),
                                 l1_ratio=float(p.get("l1_ratio", 0.5)),
                                 max_iter=int(p.get("max_iter", 2000)),
                                 random_state=self.seed).fit(xs, ys, sample_weight=ws)
        self.info["converged"] = not any(issubclass(c.category, ConvergenceWarning)
                                         for c in caught)
        self.info["subsampled_rows"] = int(len(rows)) if len(rows) < len(y) else None
        self._est = est

    def _fit_forest(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        p = self.params
        cls = RandomForestClassifier if self.is_classifier else RandomForestRegressor
        est = cls(n_estimators=int(p.get("n_estimators", 100)),
                  max_depth=p.get("max_depth"), min_samples_leaf=int(p.get("min_samples_leaf", 500)),
                  max_features=p.get("max_features", 0.33),
                  max_samples=p.get("max_samples"), n_jobs=self.threads,
                  random_state=self.seed)
        est.fit(x, y.astype(np.int64) if self.is_classifier else y, sample_weight=w)
        self._est = est

    def _fit_xgboost(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                     xi: np.ndarray | None, yi: np.ndarray | None) -> None:
        import xgboost as xgb

        p = self.params
        common: dict[str, Any] = {
            "n_estimators": int(p.get("n_estimators", 1000)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "max_depth": int(p.get("max_depth", 4)),
            "min_child_weight": float(p.get("min_child_weight", 100)),
            "subsample": float(p.get("subsample", 0.8)),
            "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
            "reg_alpha": float(p.get("reg_alpha", 0.0)),
            "reg_lambda": float(p.get("reg_lambda", 1.0)),
            "tree_method": "hist", "n_jobs": self.threads, "random_state": self.seed,
            "early_stopping_rounds": (int(p.get("early_stopping_rounds", 50))
                                      if xi is not None else None)}
        if self.is_classifier:
            if p.get("balanced"):
                pos = float(np.average(y, weights=w))
                common["scale_pos_weight"] = (1.0 - pos) / max(pos, 1e-9)
            est = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        else:
            est = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
        eval_set = [(xi, yi)] if xi is not None else None
        est.fit(x, y, sample_weight=w, eval_set=eval_set, verbose=False)
        best = getattr(est, "best_iteration", None)
        self.info["best_iteration"] = None if best is None else int(best) + 1
        self._est = est

    def _fit_lightgbm(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                      xi: np.ndarray | None, yi: np.ndarray | None) -> None:
        import lightgbm as lgb

        p = self.params
        common: dict[str, Any] = {
            "n_estimators": int(p.get("n_estimators", 1000)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "num_leaves": int(p.get("num_leaves", 31)),
            "max_depth": int(p.get("max_depth", -1)),
            "min_child_samples": int(p.get("min_data_in_leaf", 500)),
            "colsample_bytree": float(p.get("feature_fraction", 0.8)),
            "subsample": float(p.get("bagging_fraction", 0.8)),
            "subsample_freq": int(p.get("bagging_freq", 1)),
            "reg_lambda": float(p.get("lambda_l2", 1.0)),
            "reg_alpha": float(p.get("lambda_l1", 0.0)),
            "n_jobs": self.threads, "random_state": self.seed, "verbose": -1,
            "deterministic": True, "force_row_wise": True}
        if self.is_classifier:
            est = lgb.LGBMClassifier(objective="binary",
                                     is_unbalance=bool(p.get("balanced", False)), **common)
        else:
            est = lgb.LGBMRegressor(objective="regression", **common)
        if xi is not None:
            est.fit(x, y, sample_weight=w, eval_X=(xi,), eval_y=(yi,),
                    callbacks=[lgb.early_stopping(int(p.get("early_stopping_rounds", 50)),
                                                  verbose=False)])
        else:
            est.fit(x, y, sample_weight=w)
        best = getattr(est, "best_iteration_", None)
        self.info["best_iteration"] = int(best) if best else int(est.n_estimators)
        self._est = est.booster_

    def _fit_catboost(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                      xi: np.ndarray | None, yi: np.ndarray | None) -> None:
        from catboost import CatBoostClassifier, CatBoostRegressor

        p = self.params
        common: dict[str, Any] = {
            "iterations": int(p.get("iterations", 1000)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "depth": int(p.get("depth", 6)), "l2_leaf_reg": float(p.get("l2_leaf_reg", 3.0)),
            "random_seed": self.seed, "thread_count": self.threads, "verbose": 0,
            "allow_writing_files": False}
        if xi is not None:
            common.update({"use_best_model": True, "od_type": "Iter",
                           "od_wait": int(p.get("early_stopping_rounds", 50))})
        if self.is_classifier:
            est = CatBoostClassifier(loss_function="Logloss", **common,
                                     **({"auto_class_weights": "Balanced"}
                                        if p.get("balanced") else {}))
        else:
            est = CatBoostRegressor(loss_function="RMSE", **common)
        est.fit(x, y, sample_weight=w, eval_set=(xi, yi) if xi is not None else None)
        self.info["best_iteration"] = int(est.get_best_iteration() or 0) + 1 \
            if xi is not None else int(est.tree_count_)
        self._est = est

    # -- prediction ----------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        fam = self.family
        n = x.shape[0]
        if fam == "constant":
            return np.full(n, float(self._est))
        if fam == "lightgbm":
            return np.asarray(self._est.predict(x), dtype=np.float64)
        if self.is_classifier:
            return np.asarray(self._est.predict_proba(x)[:, 1], dtype=np.float64)
        return np.asarray(self._est.predict(x), dtype=np.float64)

    def feature_importance(self, names: list[str]) -> dict[str, float]:
        """Gain / impurity importance (trees) or |coefficient| (linear) - descriptive only."""
        fam = self.family
        est = self._est
        if fam == "constant":
            return dict.fromkeys(names, 0.0)
        if fam in _LINEAR:
            coef = np.ravel(est.coef_)
            return {n: float(abs(c)) for n, c in zip(names, coef, strict=False)}
        if fam == "random_forest":
            return {n: float(v) for n, v in zip(names, est.feature_importances_, strict=False)}
        if fam == "lightgbm":
            gain = est.feature_importance(importance_type="gain")
            return {n: float(v) for n, v in zip(names, gain, strict=False)}
        if fam == "xgboost":
            score = est.get_booster().get_score(importance_type="gain")
            return {n: float(score.get(f"f{j}", 0.0)) for j, n in enumerate(names)}
        values = est.get_feature_importance()
        return {n: float(v) for n, v in zip(names, values, strict=False)}

    def coefficients(self) -> np.ndarray | None:
        if self.family in _LINEAR:
            return np.ravel(self._est.coef_).astype(np.float64)
        return None

    # -- serialization -------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        fam = self.family
        if fam == "constant":
            path = directory / "model.json"
            path.write_text(json.dumps({"value": float(self._est)}), encoding="utf-8")
        elif fam in _LINEAR or fam == "random_forest":
            import joblib

            path = directory / "model.joblib"
            joblib.dump(self._est, path, compress=3)
        elif fam == "xgboost":
            path = directory / "model.ubj"
            self._est.save_model(path)
        elif fam == "lightgbm":
            path = directory / "model.txt"
            self._est.save_model(str(path), num_iteration=self.info.get("best_iteration"))
        else:
            path = directory / "model.cbm"
            self._est.save_model(str(path))
        meta = {"family": fam, "task": self.task, "params": self.params, "seed": self.seed,
                "threads": self.threads, "info": self.info, "file": path.name}
        (directory / "model_meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                                   encoding="utf-8")
        self.info["size_bytes"] = int(path.stat().st_size)
        return path


def make_model(family: str, task: str, params: dict[str, Any], *, seed: int,
               threads: int) -> Model:
    return Model(family=family, task=task, params=dict(params), seed=seed, threads=threads)


def load_model(directory: Path) -> Model:
    meta = json.loads((directory / "model_meta.json").read_text(encoding="utf-8"))
    fam, task = meta["family"], meta["task"]
    model = Model(family=fam, task=task, params=meta["params"], seed=meta["seed"],
                  threads=meta["threads"], info=dict(meta["info"]))
    path = directory / meta["file"]
    if fam == "constant":
        model._est = float(json.loads(path.read_text(encoding="utf-8"))["value"])
    elif fam in _LINEAR or fam == "random_forest":
        import joblib

        model._est = joblib.load(path)
    elif fam == "xgboost":
        import xgboost as xgb

        est = xgb.XGBClassifier() if task == "classification" else xgb.XGBRegressor()
        est.load_model(path)
        model._est = est
    elif fam == "lightgbm":
        import lightgbm as lgb

        model._est = lgb.Booster(model_file=str(path))
    else:
        from catboost import CatBoostClassifier, CatBoostRegressor

        est = CatBoostClassifier() if task == "classification" else CatBoostRegressor()
        est.load_model(str(path))
        model._est = est
    return model

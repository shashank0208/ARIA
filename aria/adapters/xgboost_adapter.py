import numpy as np
import pandas as pd
from aria.adapters.base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    """
    Adapter for XGBoost models (XGBClassifier / XGBRegressor).
    Also handles sklearn Pipelines where the final step is XGBoost.
    """

    def __init__(self, model: any, feature_names: list[str]):
        self.model = model
        self._feature_names = feature_names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self._feature_names])

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_feature_importance(self) -> dict[str, float]:
        try:
            # Handle sklearn Pipeline with XGBoost final step
            if hasattr(self.model, "named_steps"):
                xgb = list(self.model.named_steps.values())[-1]
            else:
                xgb = self.model

            importances = xgb.feature_importances_
            return dict(zip(self._feature_names, importances.tolist()))
        except Exception:
            return {f: 0.0 for f in self._feature_names}

    def get_model_type(self) -> str:
        return type(self.model).__name__


class SklearnAdapter(ModelAdapter):
    """
    Adapter for any sklearn-compatible model or Pipeline.
    Works with LogisticRegression, RandomForest, GradientBoosting, etc.
    """

    def __init__(self, model: any, feature_names: list[str]):
        self.model = model
        self._feature_names = feature_names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X[self._feature_names])
        # Regression fallback: wrap scalar predictions in 2-col array
        preds = self.model.predict(X[self._feature_names])
        return np.column_stack([1 - preds, preds])

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_feature_importance(self) -> dict[str, float]:
        try:
            if hasattr(self.model, "named_steps"):
                estimator = list(self.model.named_steps.values())[-1]
            else:
                estimator = self.model

            if hasattr(estimator, "feature_importances_"):
                return dict(zip(self._feature_names, estimator.feature_importances_.tolist()))
            elif hasattr(estimator, "coef_"):
                coefs = np.abs(estimator.coef_).flatten()
                return dict(zip(self._feature_names, coefs.tolist()))
        except Exception:
            pass
        return {f: 0.0 for f in self._feature_names}

    def get_model_type(self) -> str:
        return type(self.model).__name__


class RESTAdapter(BaseAdapter):
    def __init__(self, endpoint: str, feature_order: list[str]):
        self.endpoint = endpoint
        self.feature_order = feature_order

    def predict(self, X):
        X = X[self.feature_order]  # enforce order before serialization
        resp = requests.post(self.endpoint, json={"instances": X.to_dict("records")})
        return normalize_output(resp.json())  # same normalization every adapter must implement
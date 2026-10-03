from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class ModelAdapter(ABC):
    """
    Standard contract between ARIA agents and any ML model.
    Agents are 100% model-blind — they only call this interface.
    All framework-specific code (XGBoost, sklearn, PyTorch, etc.) lives inside
    a concrete adapter subclass. Zero leakage into agents.
    """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions. Classification: class labels. Regression: floats."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities. Shape: (n_samples, n_classes)."""
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names the model was trained on."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return {feature_name: importance_score} mapping."""
        ...

    @abstractmethod
    def get_model_type(self) -> str:
        """Human-readable model type string. E.g. 'XGBoostClassifier'."""
        ...

    def get_metrics(self, X: pd.DataFrame, y_true: np.ndarray, metrics: list[str]) -> dict[str, float]:
        """
        Compute requested metrics. Agents pass metric names from config —
        never hardcoded. Override in subclass for custom metric logic.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score, mean_squared_error, r2_score
        )

        y_pred = self.predict(X)
        results = {}

        metric_map = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "f1": lambda: f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": lambda: precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": lambda: recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": lambda: roc_auc_score(y_true, self.predict_proba(X)[:, 1]),
            "mse": lambda: mean_squared_error(y_true, y_pred),
            "r2": lambda: r2_score(y_true, y_pred),
        }

        for m in metrics:
            try:
                results[m] = float(metric_map[m]())
            except Exception as e:
                results[m] = float("nan")

        return results

"""
Utilitaires ML pour l'entrainement et la prediction de rupture a 4 semaines.

Le code utilise XGBoost si la dependance est presente. Sinon, il bascule sur
un modele logistique minimal implemente en numpy pour conserver un flux de
travail fonctionnel dans les environnements legers.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - depend de l'environnement utilisateur
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None


TARGET_COLUMN = "target_rupture_4_semaines"
DEFAULT_MODEL_NAME = "rupture_model.pkl"

FEATURE_COLUMNS = [
    "capacite_fabricant",
    "demande_lissee_fabricant",
    "disruption_fabricant",
    "duree_disruption_restante",
    "demande_patient",
    "demande_non_servie",
    "taux_service",
    "stock_grossiste",
    "stock_pharmacie",
    "rupture_grossiste",
    "rupture_pharmacie",
]


class SimpleLogisticModel:
    """Fallback simple pour garder un modele entrainable sans scikit-learn."""

    def __init__(self, learning_rate=0.05, epochs=400, l2=1e-4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.mean_ = None
        self.scale_ = None
        self.weights_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        Xs = (X - self.mean_) / self.scale_
        Xb = np.c_[np.ones(len(Xs)), Xs]
        weights = np.zeros(Xb.shape[1], dtype=float)

        for _ in range(self.epochs):
            logits = Xb @ weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            gradient = (Xb.T @ (probs - y)) / len(y)
            regularisation = np.r_[0.0, weights[1:]] * self.l2
            weights -= self.learning_rate * (gradient + regularisation)

        self.weights_ = weights
        return self

    def _prepare(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def predict_proba(self, X):
        Xs = self._prepare(X)
        Xb = np.c_[np.ones(len(Xs)), Xs]
        logits = Xb @ self.weights_
        probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _ensure_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing}")


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    _ensure_columns(df, [TARGET_COLUMN])
    return df


def prepare_matrix(df, feature_columns=None):
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = df.copy()
    missing = [col for col in feature_columns if col not in df.columns]
    for col in missing:
        df[col] = 0

    X = df[feature_columns].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    X = X.fillna(0)
    return X.astype(float).to_numpy(), list(feature_columns)


def _metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _split_train_test(X, y, test_fraction=0.2):
    split_idx = int(len(X) * (1 - test_fraction))
    split_idx = min(max(split_idx, 1), len(X) - 1 if len(X) > 1 else 1)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_model(
    dataset_path,
    model_path=None,
    feature_columns=None,
    test_fraction=0.2,
    random_state=42,
):
    df = load_dataset(dataset_path)
    feature_columns = feature_columns or FEATURE_COLUMNS

    X, used_features = prepare_matrix(df, feature_columns=feature_columns)
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_fraction=test_fraction)

    backend = "fallback-logistic"
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        backend = "xgboost"
    else:
        model = SimpleLogisticModel()
        model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = _metrics(y_test, y_prob)
    else:
        metrics = _metrics(y_train, model.predict_proba(X_train)[:, 1])

    artifact = {
        "model": model,
        "backend": backend,
        "feature_columns": used_features,
        "target_column": TARGET_COLUMN,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "metrics": metrics,
        "dataset_path": str(Path(dataset_path).resolve()),
    }

    if model_path is not None:
        save_model(artifact, model_path)

    return artifact


def save_model(artifact, model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)
    return model_path


def load_model(model_path):
    with Path(model_path).open("rb") as f:
        return pickle.load(f)


def predict_dataframe(df, artifact):
    feature_columns = artifact["feature_columns"]
    X, _ = prepare_matrix(df, feature_columns=feature_columns)
    model = artifact["model"]
    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["proba_rupture"] = proba
    result["prediction_rupture"] = (proba >= 0.5).astype(int)
    return result


def predict_single(sample, artifact):
    if isinstance(sample, pd.Series):
        sample_df = sample.to_frame().T
    elif isinstance(sample, dict):
        sample_df = pd.DataFrame([sample])
    else:
        sample_df = pd.DataFrame(sample)

    return predict_dataframe(sample_df, artifact).iloc[0].to_dict()

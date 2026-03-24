"""
src/evaluation/metrics.py

Evaluation metrics for the spectral policy learning experiments.

Three primary metrics (same triad as rl-policy-histopathology):
  - AUC   (macro OvR)
  - Accuracy
  - F1    (macro)

Plus one spectroscopy-specific metric:
  - Robustness: classification confidence under additive spectral noise
    Strengthens the instrument validity argument —
    a preprocessing policy that boosts accuracy but collapses under noise
    is not a valid measurement instrument.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from typing import Callable


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_auc(y_true, y_proba, labels=None) -> float:
    """Macro one-vs-rest AUC. Handles single-class edge case gracefully."""
    try:
        return roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=labels
        )
    except ValueError:
        return 0.0


def compute_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_robustness(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    noise_levels: tuple = (0.01, 0.05, 0.10),
    n_trials: int = 10,
    seed: int = 42,
) -> float:
    """
    Mean accuracy across additive Gaussian noise perturbations.

    Noise is scaled relative to the per-spectrum intensity range, so it is
    invariant to the absolute intensity scale (which varies with preprocessing).

    This is the instrument validity metric: a policy that achieves high
    accuracy but is sensitive to measurement noise does not produce a
    valid instrument.

    Returns
    -------
    mean_robust_accuracy : float (average over noise_levels × n_trials)
    """
    rng = np.random.default_rng(seed)
    accs = []

    intensity_range = X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
    intensity_range = np.where(intensity_range == 0, 1.0, intensity_range)

    for noise_frac in noise_levels:
        for _ in range(n_trials):
            noise = rng.standard_normal(X.shape) * noise_frac * intensity_range
            X_noisy = X + noise
            y_pred = classifier.predict(X_noisy)
            accs.append(accuracy_score(y, y_pred))

    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Reward functions (injected into SpectralEnv)
# ---------------------------------------------------------------------------

def make_reward_fn(
    metric: str,
    classifier_cls,
    classifier_kwargs: dict = None,
) -> Callable:
    """
    Factory: returns a reward function compatible with SpectralEnv.set_reward_fn().

    Parameters
    ----------
    metric          : "auc", "accuracy", or "f1"
    classifier_cls  : one of PLSDAClassifier, SVMClassifier, CNNClassifier
    classifier_kwargs : passed to classifier constructor

    Returns
    -------
    reward_fn(X_train, y_train, X_val, y_val) -> float
    """
    if classifier_kwargs is None:
        classifier_kwargs = {}

    def reward_fn(X_train, y_train, X_val, y_val):
        mask_tr  = np.isfinite(X_train).all(axis=1)
        mask_val = np.isfinite(X_val).all(axis=1)
        X_train, y_train = X_train[mask_tr], y_train[mask_tr]
        X_val,   y_val   = X_val[mask_val],  y_val[mask_val]
        if (len(X_train) < 10 or len(X_val) < 2
                or len(np.unique(y_train)) < 2
                or len(np.unique(y_val)) < 2):
            return 0.0
        try:
            clf = classifier_cls(**classifier_kwargs)
            clf.fit(X_train, y_train)
            if metric == "auc":
                proba = clf.predict_proba(X_val)
                return compute_auc(y_val, proba, labels=clf.classes_)
            elif metric == "accuracy":
                y_pred = clf.predict(X_val)
                return compute_accuracy(y_val, y_pred)
            elif metric == "f1":
                y_pred = clf.predict(X_val)
                return compute_f1(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric!r}")
        except Exception:
            return 0.0

    return reward_fn


# ---------------------------------------------------------------------------
# Full evaluation suite (called at test time, not during RL training)
# ---------------------------------------------------------------------------

def evaluate_classifier(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    compute_noise_robustness: bool = True,
) -> dict:
    """
    Run the full evaluation suite on held-out test data.

    Returns
    -------
    results : dict with keys auc, accuracy, f1, robustness (optional)
    """
    y_pred  = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    results = {
        "accuracy":  compute_accuracy(y_test, y_pred),
        "f1":        compute_f1(y_test, y_pred),
        "auc":       compute_auc(y_test, y_proba, labels=classifier.classes_),
    }

    if compute_noise_robustness:
        results["robustness"] = compute_robustness(classifier, X_test, y_test)

    return results

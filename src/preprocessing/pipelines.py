"""
src/preprocessing/pipelines.py

Discrete preprocessing pipeline action space (A0–A8).

Each pipeline is a callable: X (np.ndarray) -> X_processed (np.ndarray).

This is the spectral equivalent of the augmentation policy space in
rl-policy-histopathology. The RL agent selects a preprocessing pipeline
rather than an image augmentation sub-policy.

The current fixed pipeline in spectral-drug-verification (A6: SNV +
second derivative) serves as the static baseline condition.

Key difference from image augmentation:
  Preprocessing is applied at inference time too — the same pipeline
  must be applied to both training and test spectra. The agent therefore
  selects a pipeline that is fixed for the entire fold (train + val + test),
  not applied stochastically per sample. This makes the validity of the
  chosen pipeline more consequential than in the image augmentation case.
"""

import numpy as np
from scipy.signal import savgol_filter


# ── individual transforms ─────────────────────────────────────────────────

def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: zero-mean, unit-variance per spectrum."""
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def baseline_correction(X: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial baseline correction per spectrum."""
    n_pts = X.shape[1]
    x_pts = np.linspace(0, 1, n_pts)
    out   = np.empty_like(X)
    for i in range(len(X)):
        coeffs    = np.polyfit(x_pts, X[i], degree)
        baseline  = np.polyval(coeffs, x_pts)
        out[i]    = X[i] - baseline
    return out


def savitzky_golay_smooth(X: np.ndarray,
                           window: int = 11,
                           poly:   int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    return np.array([savgol_filter(spec, window, poly) for spec in X])


def first_derivative(X: np.ndarray,
                     window: int = 11,
                     poly:   int = 3) -> np.ndarray:
    """First derivative via Savitzky-Golay."""
    return np.array([savgol_filter(spec, window, poly, deriv=1) for spec in X])


def second_derivative(X: np.ndarray,
                      window: int = 11,
                      poly:   int = 3) -> np.ndarray:
    """Second derivative via Savitzky-Golay."""
    return np.array([savgol_filter(spec, window, poly, deriv=2) for spec in X])


def restricted_window(X: np.ndarray,
                      wavenumbers: np.ndarray,
                      low:  float = 500.0,
                      high: float = 2000.0) -> np.ndarray:
    """Restrict to a high-information wavenumber window."""
    mask = (wavenumbers >= low) & (wavenumbers <= high)
    return X[:, mask]


# ── pipeline definitions ──────────────────────────────────────────────────

def make_pipelines(wavenumbers: np.ndarray) -> dict:
    """
    Return the action-space dictionary: {action_id: pipeline_fn}.

    Each pipeline_fn takes X (n_spectra, n_wavenumbers) and returns
    X_processed with the same or reduced number of features.

    Note: restricted_window (A8) changes the feature dimension.
    Models trained with A8 cannot be evaluated with other pipelines —
    the RL agent treats this as a separate action and retrains the
    classifier entirely under that pipeline.
    """
    return {
        0: lambda X: X,                                         # A0 raw
        1: lambda X: snv(X),                                    # A1 SNV
        2: lambda X: baseline_correction(X),                    # A2 baseline
        3: lambda X: savitzky_golay_smooth(X),                  # A3 SG smooth
        4: lambda X: first_derivative(X),                       # A4 1st deriv
        5: lambda X: second_derivative(X),                      # A5 2nd deriv
        6: lambda X: second_derivative(snv(X)),                 # A6 SNV+2nd [static baseline]
        7: lambda X: savitzky_golay_smooth(snv(X)),             # A7 SNV+SG
        8: lambda X: restricted_window(X, wavenumbers),         # A8 window
    }


N_ACTIONS          = 9
STATIC_BASELINE_ID = 6

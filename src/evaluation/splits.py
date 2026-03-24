"""
src/evaluation/splits.py

The methodological core of the paper.

Two split strategies:
  - "acquisition"  : correct split — spectra from the same acquisition group
                     are never split across train/val/test.
                     Corrects the field-wide leakage error.
  - "random"       : leaky baseline — random spectrum-level split.
                     Reproduces the error common in Raman ML literature.

The dataset has no explicit acquisition ID column.
We reconstruct acquisition groups from consecutive same-label runs —
consistent with how Raman datasets are collected (all spectra from one
measurement session are stored contiguously).
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Acquisition group inference
# ---------------------------------------------------------------------------

def infer_acquisition_groups(labels: np.ndarray) -> np.ndarray:
    """
    Assign an acquisition group ID to each spectrum based on consecutive
    same-label runs.

    Spectra collected in the same session appear as contiguous blocks of the
    same compound label. A new group starts whenever the label changes.

    Returns
    -------
    group_ids : np.ndarray of int, shape (n_spectra,)
    """
    group_ids = np.zeros(len(labels), dtype=int)
    current_group = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            current_group += 1
        group_ids[i] = current_group
    return group_ids


# ---------------------------------------------------------------------------
# Core split functions
# ---------------------------------------------------------------------------

def acquisition_split(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
]:
    """
    Acquisition-level split: all spectra from the same acquisition group
    are assigned to the same partition.

    This is the CORRECT split strategy. It prevents within-acquisition
    correlation from leaking between train/val/test — the error we are
    correcting in this paper.

    Splits are stratified by compound label at the group level.
    """
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(group_ids)
    # Label for each group = majority label in that group
    group_labels = np.array([
        pd.Series(y[group_ids == g]).mode()[0] for g in unique_groups
    ])

    unique_labels = np.unique(group_labels)
    val_groups, test_groups = [], []

    for label in unique_labels:
        label_groups = unique_groups[group_labels == label]
        n = len(label_groups)
        shuffled = rng.permutation(label_groups)

        n_test = max(1, int(np.ceil(n * test_frac)))
        n_val  = max(1, int(np.ceil(n * val_frac)))

        test_groups.extend(shuffled[:n_test].tolist())
        val_groups.extend(shuffled[n_test:n_test + n_val].tolist())

    test_set = set(test_groups)
    val_set  = set(val_groups)

    test_mask  = np.isin(group_ids, list(test_set))
    val_mask   = np.isin(group_ids, list(val_set))
    train_mask = ~test_mask & ~val_mask

    return (
        X[train_mask], y[train_mask],
        X[val_mask],   y[val_mask],
        X[test_mask],  y[test_mask],
    )


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Random spectrum-level split.

    This is the LEAKY BASELINE — it reproduces the methodological error
    common in Raman ML literature. Spectra from the same acquisition session
    can appear in both train and test, inflating performance estimates.

    We include this explicitly to quantify the leakage effect (Table 1).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)

    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


# ---------------------------------------------------------------------------
# Unified interface used by run_experiment.py
# ---------------------------------------------------------------------------

def get_splits(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "acquisition",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Parameters
    ----------
    strategy : "acquisition" (correct) or "random" (leaky baseline)

    Returns
    -------
    splits : dict with keys X_train, y_train, X_val, y_val, X_test, y_test
    info   : dict with split sizes and group counts (acquisition only)
    """
    if strategy == "acquisition":
        group_ids = infer_acquisition_groups(y)
        n_groups = len(np.unique(group_ids))
        X_tr, y_tr, X_val, y_val, X_te, y_te = acquisition_split(
            X, y, group_ids, val_frac, test_frac, seed
        )
        info = {
            "strategy": "acquisition",
            "n_acquisition_groups": n_groups,
            "n_train": len(y_tr),
            "n_val": len(y_val),
            "n_test": len(y_te),
        }
    elif strategy == "random":
        X_tr, y_tr, X_val, y_val, X_te, y_te = random_split(
            X, y, val_frac, test_frac, seed
        )
        info = {
            "strategy": "random",
            "n_train": len(y_tr),
            "n_val": len(y_val),
            "n_test": len(y_te),
        }
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}. "
                         f"Use 'acquisition' or 'random'.")

    splits = {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_val, "y_val":  y_val,
        "X_test":  X_te,  "y_test": y_te,
    }
    return splits, info

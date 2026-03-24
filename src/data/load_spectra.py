"""
src/data/load_spectra.py

Load Raman spectra from CSV.

Expected format (matches spectral-drug-verification):
  - Columns: spectral wavenumber values (e.g. 150.0, 151.0, ... 3425.0)
  - One column named 'label' (compound identity string)
  - No patient/subject ID — each spectrum is independent
"""

import numpy as np
import pandas as pd


def load_spectra(dataset_path: str,
                 label_column: str = "label",
                 wavenumber_min: float = 150.0,
                 wavenumber_max: float = 3425.0) -> tuple:
    """
    Load spectra CSV and return (X, y, wavenumbers, class_names).

    Returns
    -------
    X           : np.ndarray shape (n_spectra, n_wavenumbers)
    y           : np.ndarray shape (n_spectra,) — integer class labels
    wavenumbers : np.ndarray of wavenumber values (float)
    class_names : list of str — maps integer label -> compound name
    """
    df = pd.read_csv(dataset_path)

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found.\n"
            f"Available columns: {list(df.columns[:5])} ..."
        )

    # Identify spectral columns (numeric column names within wavenumber range)
    spectral_cols = []
    for col in df.columns:
        try:
            wn = float(col)
            if wavenumber_min <= wn <= wavenumber_max:
                spectral_cols.append(col)
        except ValueError:
            continue

    if not spectral_cols:
        raise ValueError(
            f"No spectral columns found in range "
            f"[{wavenumber_min}, {wavenumber_max}].\n"
            f"Column sample: {list(df.columns[:5])}"
        )

    wavenumbers = np.array([float(c) for c in spectral_cols])
    X_raw       = df[spectral_cols].values.astype(np.float32)
    labels_raw  = df[label_column].values

    # Encode string labels to integers
    class_names = sorted(set(labels_raw))
    label_map   = {name: i for i, name in enumerate(class_names)}
    y           = np.array([label_map[l] for l in labels_raw], dtype=np.int32)

    n_classes = len(class_names)
    print(f"Loaded {len(X_raw):,} spectra  "
          f"{len(spectral_cols):,} wavenumbers  "
          f"{n_classes} compounds")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"  [{i}] {name}: {count} spectra")

    return X_raw, y, wavenumbers, class_names

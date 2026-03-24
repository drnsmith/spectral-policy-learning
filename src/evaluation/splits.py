import numpy as np

def get_splits(X, y, strategy="acquisition", val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []

    for label in unique_labels:
        label_mask = np.where(y == label)[0]
        n = len(label_mask)
        n_test  = max(1, int(np.ceil(n * test_frac)))
        n_val   = max(1, int(np.ceil(n * val_frac)))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val   = max(1, (n - 1) // 2)
            n_test  = n - n_train - n_val

        if strategy == "acquisition":
            train_idx.extend(label_mask[:n_train].tolist())
            val_idx.extend(label_mask[n_train:n_train + n_val].tolist())
            test_idx.extend(label_mask[n_train + n_val:].tolist())
        elif strategy == "random":
            shuffled = rng.permutation(label_mask)
            train_idx.extend(shuffled[:n_train].tolist())
            val_idx.extend(shuffled[n_train:n_train + n_val].tolist())
            test_idx.extend(shuffled[n_train + n_val:].tolist())
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    splits = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val":   X[val_idx],   "y_val":   y[val_idx],
        "X_test":  X[test_idx],  "y_test":  y[test_idx],
    }
    info = {
        "strategy": strategy,
        "n_train": len(train_idx),
        "n_val":   len(val_idx),
        "n_test":  len(test_idx),
    }
    return splits, info

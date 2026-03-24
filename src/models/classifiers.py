"""
src/models/classifiers.py

Thin wrappers for PLS-DA, SVM, and 1D-CNN with a common interface.
Classifiers are the vessel — not the contribution.
The same interface lets run_experiment.py loop over all three cleanly.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# PLS-DA
# ---------------------------------------------------------------------------

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS Discriminant Analysis.
    Encodes labels → one-hot, fits PLS regression, argmax for prediction.
    Standard in Raman / analytical chemistry — credibility anchor.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.le = LabelEncoder()
        self.pls = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self.le.fit_transform(y).astype(int)
        n_classes = len(self.le.classes_)
        # One-hot
        Y = np.eye(n_classes)[y_enc]
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls = PLSRegression(n_components=n_comp)
        self.pls.fit(X, Y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.pls.predict(X)
        # Softmax to get probabilities
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))

    @property
    def classes_(self):
        return self.le.classes_


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    RBF-SVM with probability calibration.
    Efficient on moderate-n spectral data.
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf"):
        self.C = C
        self.kernel = kernel
        self.model = SVC(C=C, kernel=kernel, probability=True)
        self.le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self.le.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self.model.predict(X)
        return self.le.inverse_transform(y_enc)

    @property
    def classes_(self):
        return self.le.classes_


# ---------------------------------------------------------------------------
# 1D-CNN  (PyTorch — mirrors Repo 1's deep learning approach)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _CNN1D(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1D-CNN for spectral classification (PyTorch).
    Falls back to a warning if torch is not installed.
    """

    def __init__(
        self,
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device
        self.le = LabelEncoder()
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self.le.fit_transform(y)
        n_classes = len(self.le.classes_)

        self.model = _CNN1D(X.shape[1], n_classes).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        y_t = torch.LongTensor(y_enc).to(self.device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.n_epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).unsqueeze(1).to(self.device)
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))

    @property
    def classes_(self):
        return self.le.classes_


# ---------------------------------------------------------------------------
# Registry — used by run_experiment.py
# ---------------------------------------------------------------------------

CLASSIFIERS = {
    "plsda": PLSDAClassifier,
    "svm":   SVMClassifier,
    "cnn":   CNNClassifier,
}

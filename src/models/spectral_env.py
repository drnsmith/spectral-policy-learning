"""
src/models/spectral_env.py

RL environment for Raman spectral preprocessing policy learning.
State: current spectrum (after partial preprocessing)
Actions: 7 preprocessing sub-policies
Reward: validation metric from downstream classifier
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Preprocessing sub-policies
# ---------------------------------------------------------------------------

def _savitzky_golay(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing (window=11, poly=3)."""
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window_length=11, polyorder=3), axis=1, arr=X
    )


def _als_baseline(X: np.ndarray, lam: float = 1e5, p: float = 0.01,
                  n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline correction."""
    def _als_single(y):
        n = len(y)
        D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).toarray()
        D = lam * D.T @ D
        w = np.ones(n)
        for _ in range(n_iter):
            W = diags(w, 0)
            Z = np.linalg.solve(W + D, w * y)
            w = p * (y > Z) + (1 - p) * (y <= Z)
        return y - Z
    return np.apply_along_axis(_als_single, axis=1, arr=X)


def _snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate normalisation."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def _first_derivative(X: np.ndarray) -> np.ndarray:
    """First derivative (Norris-Williams, gap=1)."""
    return np.diff(X, n=1, axis=1, prepend=X[:, :1])


def _second_derivative(X: np.ndarray) -> np.ndarray:
    """Second derivative."""
    return np.diff(X, n=2, axis=1, prepend=X[:, :2])


def _minmax(X: np.ndarray) -> np.ndarray:
    """Min-max normalisation per spectrum."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)


def _noop(X: np.ndarray) -> np.ndarray:
    """Identity — no preprocessing."""
    return X.copy()


POLICIES = {
    0: ("savitzky_golay",    _savitzky_golay),
    1: ("als_baseline",      _als_baseline),
    2: ("snv",               _snv),
    3: ("first_derivative",  _first_derivative),
    4: ("second_derivative", _second_derivative),
    5: ("minmax",            _minmax),
    6: ("noop",              _noop),
}

N_ACTIONS = len(POLICIES)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SpectralEnv:
    """
    RL environment for sequential spectral preprocessing.

    Episode flow:
      reset()  → initial state (raw spectra batch)
      step(a)  → apply sub-policy a, return (next_state, reward, done)

    State:  flat 1-D feature vector = mean spectrum of current X_train batch
            (summary statistic so state dim stays fixed regardless of batch size)
    Reward: validation metric from downstream classifier (set externally via
            .set_reward_fn(fn) where fn(X_train, y_train, X_val, y_val) → float)
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_steps: int = 5,
    ):
        self.X_train_raw = X_train.copy()
        self.y_train = y_train
        self.X_val_raw = X_val.copy()
        self.y_val = y_val
        self.max_steps = max_steps

        self.reward_fn = None          # injected before training
        self.n_features = X_train.shape[1]
        self.state_dim = self.n_features

        self._reset_state()

    # ------------------------------------------------------------------
    def set_reward_fn(self, fn):
        """Inject reward function: fn(X_tr, y_tr, X_val, y_val) -> float."""
        self.reward_fn = fn

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_state()

    def _reset_state(self):
        self.X_train_current = self.X_train_raw.copy()
        self.X_val_current = self.X_val_raw.copy()
        self.step_count = 0
        self.action_history = []
        self.done = False

    def _get_state(self) -> np.ndarray:
        """Mean spectrum of current training batch (fixed-size state vector)."""
        return self.X_train_current.mean(axis=0)

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Apply sub-policy `action` to current spectra.

        Returns
        -------
        next_state : np.ndarray
        reward     : float
        done       : bool
        info       : dict
        """
        assert not self.done, "Call reset() before stepping."
        assert action in POLICIES, f"Invalid action {action}"

        policy_name, policy_fn = POLICIES[action]

        # Apply to both train and val so evaluation is consistent
        self.X_train_current = policy_fn(self.X_train_current)
        self.X_val_current   = policy_fn(self.X_val_current)
        self.action_history.append(policy_name)
        self.step_count += 1

        # Reward
        if self.reward_fn is None:
            raise RuntimeError("Set a reward function with .set_reward_fn()")
        reward = self.reward_fn(
            self.X_train_current, self.y_train,
            self.X_val_current,   self.y_val,
        )

        self.done = self.step_count >= self.max_steps
        next_state = self._get_state()

        info = {
            "step": self.step_count,
            "action_name": policy_name,
            "action_history": list(self.action_history),
            "reward": reward,
        }
        return next_state, reward, self.done, info

    # ------------------------------------------------------------------
    @property
    def processed_train(self) -> np.ndarray:
        return self.X_train_current

    @property
    def processed_val(self) -> np.ndarray:
        return self.X_val_current

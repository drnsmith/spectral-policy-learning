"""
src/rl/agent.py

Same Q-learning agent as rl-policy-histopathology, adapted for
multiclass spectral classification.

State  : (last_action_id, metric_bucket)
           metric_bucket 0 = low  (<0.85)
                         1 = mid  (0.85–0.95)
                         2 = high (>0.95)
Action : int in {0 … N_ACTIONS-1}  — preprocessing pipeline
Reward : delta_metric = val_metric_t − val_metric_{t−1}

Reward variants:
  "macro_f1"          — recommended (accounts for class imbalance)
  "balanced_accuracy" — alternative
  "roc_auc_ovr"       — one-vs-rest AUC for multiclass
"""

import numpy as np


def _metric_bucket(val: float, low: float, high: float) -> int:
    if val < low:   return 0
    if val < high:  return 1
    return 2


def encode_state(action_id: int, metric: float,
                 low: float = 0.85, high: float = 0.95) -> tuple:
    return (action_id, _metric_bucket(metric, low, high))


class QLearningAgent:
    def __init__(self, n_actions=9, alpha=0.1, gamma=0.9,
                 epsilon_start=0.3, epsilon_end=0.05,
                 epsilon_decay=0.92, seed=42):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)
        self.Q:       dict = {}
        self._history:list = []

    def _q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.Q[state]

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self._q(state)))

    def update(self, state, action, reward, next_state):
        q_sa   = self._q(state)[action]
        q_next = np.max(self._q(next_state))
        self._q(state)[action] += self.alpha * (
            reward + self.gamma * q_next - q_sa)

    def step_epsilon(self):
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def record(self, action, metric):
        self._history.append((action, metric))

    def best_action(self):
        if not self._history:
            return STATIC_BASELINE_ID
        return max(self._history, key=lambda t: t[1])[0]

    def reset(self, seed=None):
        self.Q        = {}
        self._history = []
        if seed is not None:
            self.rng = np.random.default_rng(seed)


STATIC_BASELINE_ID = 6

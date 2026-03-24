"""
src/models/q_agent.py

Tabular-ish Q-learning agent with discretised state bins.
Mirrors the Q-learning approach in rl-policy-histopathology for
cross-domain methodological consistency.
"""

import numpy as np
import json
from pathlib import Path


class QAgent:
    """
    Q-learning agent for discrete action selection over spectral sub-policies.

    State is the mean spectrum (3276-dim). We discretise it to a compact
    key via PCA projection + sign-binarisation so a table stays tractable.
    For reproducibility the projection matrix is fixed at init from a seed.

    Parameters
    ----------
    n_actions   : number of sub-policies (7)
    state_dim   : raw state dimension (3276)
    n_bins      : PCA components used for state hashing (default 16)
    lr          : learning rate
    gamma       : discount factor
    epsilon     : initial exploration rate
    epsilon_min : floor for epsilon
    epsilon_decay: multiplicative decay per episode
    seed        : RNG seed
    """

    def __init__(
        self,
        n_actions: int = 7,
        state_dim: int = 3276,
        n_bins: int = 16,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Fixed random projection for state hashing (no fitting needed)
        self.proj = self.rng.standard_normal((state_dim, n_bins))
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-8

        self.q_table: dict[str, np.ndarray] = {}
        self.episode_rewards: list[float] = []
        self.episode_actions: list[list[int]] = []

    # ------------------------------------------------------------------
    def _hash_state(self, state: np.ndarray) -> str:
        """Project state → n_bins dimensions → binarise → string key."""
        projected = state @ self.proj          # (n_bins,)
        binary = (projected > 0).astype(int)
        return "".join(map(str, binary))

    def _get_q(self, state_key: str) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        key = self._hash_state(state)
        return int(np.argmax(self._get_q(key)))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Bellman update."""
        s_key  = self._hash_state(state)
        ns_key = self._hash_state(next_state)

        q_current = self._get_q(s_key)[action]
        q_next    = 0.0 if done else np.max(self._get_q(ns_key))
        td_target = reward + self.gamma * q_next
        self._get_q(s_key)[action] += self.lr * (td_target - q_current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def record_episode(self, total_reward: float, actions: list[int]):
        self.episode_rewards.append(total_reward)
        self.episode_actions.append(actions)

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Serialise Q-table and metadata to JSON."""
        data = {
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_actions": self.episode_actions,
            "config": {
                "n_actions": self.n_actions,
                "state_dim": self.state_dim,
                "n_bins": self.n_bins,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Agent saved → {path}")

    @classmethod
    def load(cls, path: str) -> "QAgent":
        with open(path) as f:
            data = json.load(f)
        cfg = data["config"]
        agent = cls(**cfg)
        agent.q_table = {k: np.array(v) for k, v in data["q_table"].items()}
        agent.epsilon = data["epsilon"]
        agent.episode_rewards = data["episode_rewards"]
        agent.episode_actions = data["episode_actions"]
        return agent

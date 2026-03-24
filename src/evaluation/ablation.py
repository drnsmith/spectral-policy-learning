"""
src/evaluation/ablation.py

Ablation studies for the spectral policy learning experiments.

Two ablation axes (the paper's two claims):
  1. Split strategy: acquisition-level vs. random (Table 1)
     — quantifies the leakage effect
  2. Reward metric: AUC vs. accuracy vs. F1 (Table 2)
     — reward metric as instrument validity problem
"""

import numpy as np
import json
import time
from pathlib import Path
from itertools import product

from src.evaluation.metrics import make_reward_fn, evaluate_classifier
from src.evaluation.splits import get_splits
from src.models.spectral_env import SpectralEnv
from src.models.q_agent import QAgent
from src.models.classifiers import CLASSIFIERS


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_single(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    reward_metric: str,
    split_strategy: str,
    n_episodes: int = 200,
    max_steps: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run one full experiment condition.

    Parameters
    ----------
    classifier_name : "plsda", "svm", or "cnn"
    reward_metric   : "auc", "accuracy", or "f1"
    split_strategy  : "acquisition" (correct) or "random" (leaky)
    n_episodes      : RL training episodes
    max_steps       : sub-policy steps per episode
    seed            : RNG seed for reproducibility

    Returns
    -------
    result : dict with config, training curve, final test metrics
    """
    t0 = time.time()

    # --- Splits ---
    splits, split_info = get_splits(X, y, strategy=split_strategy, seed=seed)
    X_train = splits["X_train"];  y_train = splits["y_train"]
    X_val   = splits["X_val"];    y_val   = splits["y_val"]
    X_test  = splits["X_test"];   y_test  = splits["y_test"]

    # --- Classifier and reward function ---
    clf_cls = CLASSIFIERS[classifier_name]
    reward_fn = make_reward_fn(reward_metric, clf_cls)

    # --- RL environment and agent ---
    env = SpectralEnv(X_train, y_train, X_val, y_val, max_steps=max_steps)
    env.set_reward_fn(reward_fn)
    agent = QAgent(seed=seed)

    # --- Training loop ---
    episode_rewards = []
    best_policy = None
    best_reward = -np.inf

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        actions_taken = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            actions_taken.append(action)

        agent.decay_epsilon()
        agent.record_episode(total_reward, actions_taken)
        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = list(info["action_history"])

        if verbose and (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"reward={total_reward:.4f} | ε={agent.epsilon:.3f}")

    # --- Final evaluation: apply best policy to test set ---
    # Re-apply the best discovered policy from scratch on train+test
    X_tr_proc = X_train.copy()
    X_te_proc  = X_test.copy()

    from src.models.spectral_env import POLICIES as _POLICIES
    policy_name_to_fn = {v[0]: v[1] for v in _POLICIES.values()}

    for policy_name in best_policy:
        fn = policy_name_to_fn[policy_name]
        X_tr_proc = fn(X_tr_proc)
        X_te_proc  = fn(X_te_proc)

    final_clf = clf_cls()
    final_clf.fit(X_tr_proc, y_train)
    test_metrics = evaluate_classifier(final_clf, X_te_proc, y_test)

    elapsed = time.time() - t0

    result = {
        "config": {
            "classifier": classifier_name,
            "reward_metric": reward_metric,
            "split_strategy": split_strategy,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "seed": seed,
        },
        "split_info": split_info,
        "training": {
            "episode_rewards": episode_rewards,
            "best_policy": best_policy,
            "best_episode_reward": best_reward,
        },
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
    }

    if verbose:
        print(f"\n  ✓ {classifier_name} | {reward_metric} | {split_strategy}")
        print(f"    Best policy: {best_policy}")
        print(f"    Test metrics: {test_metrics}")
        print(f"    Time: {elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Full ablation grid
# ---------------------------------------------------------------------------

def run_ablation_grid(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: list = None,
    reward_metrics: list = None,
    split_strategies: list = None,
    n_episodes: int = 200,
    max_steps: int = 5,
    seed: int = 42,
    output_dir: str = "results/",
) -> list[dict]:
    """
    Run the full ablation grid: classifiers × reward_metrics × split_strategies.

    Default grid:
      classifiers     : ["plsda", "svm", "cnn"]
      reward_metrics  : ["auc", "accuracy", "f1"]
      split_strategies: ["acquisition", "random"]

    Total conditions: 3 × 3 × 2 = 18
    """
    if classifiers is None:
        classifiers = ["plsda", "svm", "cnn"]
    if reward_metrics is None:
        reward_metrics = ["auc", "accuracy", "f1"]
    if split_strategies is None:
        split_strategies = ["acquisition", "random"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []
    total = len(classifiers) * len(reward_metrics) * len(split_strategies)

    print(f"\nRunning ablation grid: {total} conditions")
    print(f"  Classifiers:      {classifiers}")
    print(f"  Reward metrics:   {reward_metrics}")
    print(f"  Split strategies: {split_strategies}")
    print("=" * 60)

    for i, (clf, rm, ss) in enumerate(
        product(classifiers, reward_metrics, split_strategies), 1
    ):
        print(f"\n[{i}/{total}] {clf} | {rm} reward | {ss} split")
        result = run_single(
            X, y,
            classifier_name=clf,
            reward_metric=rm,
            split_strategy=ss,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        all_results.append(result)

        # Save incrementally
        fname = f"{output_dir}/{clf}_{rm}_{ss}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save full grid
    grid_path = f"{output_dir}/ablation_grid.json"
    with open(grid_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Full grid saved → {grid_path}")

    return all_results

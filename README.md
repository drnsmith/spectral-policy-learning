# Adaptive preprocessing policy search via reinforcement learning for noisy multiclass Raman spectral classification.

---

## What this is

This is stage 2 in a two-stages project on adaptive policy learning for
noisy high-stakes measurement systems. Stage 1
([rl-policy-histopathology](https://github.com/drnsmith/rl-policy-histopathology))
established the framework for image augmentation in histopathology.

This project re-instantiates the same framework with a different policy
object: instead of selecting an image augmentation sub-policy, the RL
agent selects a **spectral preprocessing pipeline** from a discrete
action space (SNV, derivatives, baseline correction, etc.).

The central argument is the same as stage 1: the choice of reward
metric used to guide the RL agent is a **measurement validity problem**.
In a multiclass drug verification setting, macro-F1 is argued to be
the most valid reward instrument because it weights all compound
misidentifications equally — consistent with the operational objective
where all identification errors carry similar risk.

---

## Relationship to spectral-drug-verification

This repo extends
[spectral-drug-verification](https://github.com/drnsmith/spectral-drug-verification)
by replacing its fixed preprocessing pipeline (SNV + second derivative)
with an RL-guided adaptive policy. The static pipeline becomes the
baseline experimental condition.

---

## Status

Experiments in progress. Results pending.

---

## Setup

```bash
git clone https://github.com/drnsmith/spectral-policy-learning.git
cd spectral-policy-learning
pip install -r requirements.txt
```

Place your Raman spectra CSV at `data/raw/raman_spectra_api_compounds.csv`
(or update `configs/config.yaml` with your actual path).

---

## Running experiments

```bash
# All conditions
python scripts/run_experiment.py

# Specific conditions
python scripts/run_experiment.py --conditions static rl_macro_f1 rl_auc
```

---

## Experimental conditions

| Condition | Description |
|---|---|
| `none` | No preprocessing (raw spectra) |
| `static` | Fixed A6: SNV + 2nd derivative (spectral-drug-verification default) |
| `rl_macro_f1` | RL-guided, reward = Δmacro-F1 *(recommended)* |
| `rl_balanced_acc` | RL-guided, reward = Δbalanced accuracy |
| `rl_auc` | RL-guided, reward = ΔOvR-AUC |

---

## Preprocessing action space

| Action | Pipeline |
|---|---|
| A0 | Raw |
| A1 | SNV |
| A2 | Baseline correction |
| A3 | Savitzky-Golay smoothing |
| A4 | First derivative |
| A5 | Second derivative |
| A6 | SNV + second derivative *(static baseline)* |
| A7 | SNV + SG smoothing |
| A8 | Restricted wavenumber window |

---

## Paper

*Adaptive Preprocessing Policy Search via Reinforcement Learning for
Noisy Multiclass Spectral Classification* (preprint forthcoming).


---

## License

MIT — see [LICENSE](LICENSE).

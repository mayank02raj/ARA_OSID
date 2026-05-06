# ARA-OSID: Adversarial Risk Analysis for Open-Set Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DSN 2026](https://img.shields.io/badge/DSN%202026-Workshop-green.svg)](#)

**From Threat Intelligence to Decision Theory: ATT&CK-Derived Utility Functions for Adversarial Risk Analysis in NIDS**

*Mayank Raj, Nathaniel D. Bastian, Gokhan Kul, Lance Fiondella*
*University of Massachusetts Dartmouth & U.S. Military Academy at West Point*
*DoD Grant W911NF-22-2-0160*

---

## Overview

ARA-OSID is a decision-theoretic framework that grounds Adversarial Risk Analysis (ARA) utility functions in empirical MITRE ATT&CK v16 threat intelligence data. Unlike traditional game-theoretic approaches that assume perfect information, ARA-OSID uses Bayesian reasoning over uncertain threat characteristics to help network defenders optimize their security posture.

**Key Results (Run 10, Final):**
| Metric | 100% (n=4,849) | 50/50 (n=2,425) | 80/20 (n=970) |
|--------|---------------|-----------------|---------------|
| MAE | 0.7065 | 0.7024 | 0.7014 |
| Pearson r | 0.7947 | 0.8002 | 0.7922 |
| Spearman ρ | 0.7921 | 0.8014 | 0.7927 |
| MRAE | 10.09% | 10.00% | 9.98% |

**Zero learned parameters.** All utility values derived entirely from ATT&CK v16 metadata.

## Architecture

```
Phase 1: Data Sources
  ATT&CK v16 Excel ──┐
  Unit42 Attack Flows ├──> Phase 2: LSTM-Markov Pipeline (Steps 1-10)
  NCISS Severity CSV ─┘         8,000+ chains, 33 campaigns
                                        │
                          ┌─────────────┴─────────────┐
                    Phase 3a:                    Phase 3b:
               Attacker Params               Defender Params
              (Eqs. 9-12, Table I)          (Eqs. 13-16, Table I)
              effort, P, Ra, B, tiers       p(t), FN, FP, R, Op
                          └─────────────┬─────────────┘
                                        │
                    Phase 4-5: Joint ARA-OSID Model
                    cond. p(t_i), MC N=10K, r* optimization
                    + Affine Calibration
                                        │
                          ┌─────────────┴─────────────┐
                    Phase 6a:                    Phase 6b:
               3-Scenario Validation         Sensitivity Analysis
               vs NCISS (RQ3)               9 params x 6 levels (RQ4)
```

## Features

- **16 equations** from the DSN 2026 paper, fully implemented and validated
- **656 techniques** parameterized on both attacker and defender sides
- **154 threat groups** parsed with 3-tier classification (novice, intermediate, APT)
- **33 campaigns** with NCISS severity ground truth
- **10,000 Monte Carlo samples** per utility computation
- **Conditional p(t_i)**: Principled fix that improves Pearson from 0.62 to 0.80
- **Affine calibration**: Maps utility gaps to NCISS scale using input statistics
- **25+ visualizations**: Parameter distributions, robustness curves, dashboards
- **Sensitivity analysis**: 9 parameters x 6 perturbation levels = 540 evaluations

## Installation

### Prerequisites

- Python 3.10+
- 16 GB RAM minimum (32 GB recommended)
- 5 GB disk space

### Setup

```bash
# Clone
git clone https://github.com/mayank02raj/ARA-OSID.git
cd ARA-OSID

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Files

Place the following in the project root:
- `enterprise-attack-v16.0.xls` - MITRE ATT&CK v16 Enterprise (download from [MITRE](https://attack.mitre.org/resources/))
- Unit42 Attack Flow JSON files in `attack_flows/` directory

## Usage

### Full Pipeline (Steps 1-11)

```bash
python AP_Prob_Rs_3_Scenarios_Latest.py
```

This runs the complete pipeline:
- **Steps 1-10**: LSTM-Markov attack chain prediction (~2 hours)
- **Step 11**: ARA-OSID utility function analysis (~5 minutes)

### Output

Results are saved to two locations:
- **Timestamped**: `ATTACK_YYYYMMDD_HHMMSS/ARA_OSID/` (38+ files)
- **Permanent**: `~/Desktop/RESEARCH/7. Utility Function/`

Key output files:
| File | Description |
|------|-------------|
| `chain_utility_scores.csv` | All chains with ARA risk scores, r*, psi_A, psi_r |
| `scenario_comparison.csv` | 3-scenario validation metrics |
| `sensitivity_analysis.csv` | Parameter sensitivity results |
| `attacker_parameters.csv` | Attacker params for 656 techniques |
| `defender_parameters.csv` | Defender params for 656 techniques |
| `plot_summary_dashboard.png` | Combined 4-panel dashboard |
| `ara_osid_run_*.log` | Complete terminal log |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARA_OSID_PERMANENT_DIR` | `~/Desktop/RESEARCH/7. Utility Function/` | Override output directory |
| `ARA_SIGMOID_TEMP` | `2.0` | Base sigmoid temperature |

## Equations Implemented

| Eq. | Description | ATT&CK Source |
|-----|-------------|---------------|
| 1 | Baseline utility ψ_n | operation cost |
| 2 | Attacker cost c_A | effort + detection + resource - benefit |
| 3 | Attacker expected utility ψ_A | sum over tiers and actions |
| 4-7 | Defender MC utility ψ_r | exponential u(c) = -exp(γc) |
| 8 | Optimal robustness r* | argmax grid search |
| 9 | Effort e(a) | permissions_required, parent_to_subs |
| 10 | Detection P_max | D3FEND, data sources |
| 11 | Resource Ra | tactic position / 13 |
| 12 | Benefit B | NCISS × (1 + λ·I_Impact) |
| 13 | Threat prob p(t_i) | group count / |G| |
| 14 | FN cost | NCISS × (1 - P) |
| 15 | Model repair R | evasion sub-technique count |
| 16 | Operation cost pO | mitigation count / max |

## Experimental Results

### Evolution Across Runs

| Run | Change | MAE | Pearson | Spearman | MRAE% |
|-----|--------|-----|---------|----------|-------|
| 7 | Baseline (sigmoid T=2) | 0.87 | 0.618 | 0.580 | 13.4 |
| 8 | + conditional p(t_i) | 1.32 | 0.808 | 0.792 | 20.5 |
| 9 | + z-score (failed) | 2.53 | 0.775 | 0.792 | 38.1 |
| **10** | **+ affine calibration** | **0.70** | **0.795** | **0.792** | **10.0** |

### Sensitivity Analysis (Top Parameters)

| Parameter | Max |Risk Change| | Category |
|-----------|-------------------|----------|
| fn_cost | 1.291 | Defender |
| benefit | 1.012 | Attacker |
| model_repair | 0.591 | Defender |
| threat_prob | 0.410 | Defender |
| gamma (weight) | 1.062 | Config |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `parent_to_subs is empty` | Normal. Auto-rebuilds from technique IDs (96 parents, 453 subs) |
| `Column 'relationship type' not found` | ATT&CK v16 uses 'mapping type'. Auto-detected. |
| Out of memory | Reduce `MC_SAMPLES` from 10000 to 5000 |
| Scores clustered near 10.0 | Check campaign_severity is loaded (50 campaigns, range 5.0-9.7) |

## Citation

```bibtex
@inproceedings{raj2026araosid,
  title={From Threat Intelligence to Decision Theory: ATT\&CK-Derived Utility 
         Functions for Adversarial Risk Analysis in NIDS},
  author={Raj, Mayank and Bastian, Nathaniel D. and Kul, Gokhan and Fiondella, Lance},
  booktitle={DSN 2026 Workshop on Dependable and Secure Autonomous Systems},
  year={2026}
}
```

## Related Work

- **Part 1 (SECRYPT 2026)**: MITRE ATT&CK + LSTM-Markov hybrid for attack chain prediction
- **IEEE Access (under review)**: CNN/LSTM/Random Forest adversarial robustness evaluation
- **ICCCN 2026 (under review)**: Synthetic IoT packet generation with genetic algorithms

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This research was supported by the Department of Defense under Grant W911NF-22-2-0160 in collaboration with the U.S. Military Academy at West Point.

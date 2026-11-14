# Model findings (written by notebooks/04_MODELLING.ipynb)

Trained 2026-11-12 12:37. Test = rows dated 8-10 Sept, weighted (estimates on the full population).

## Headline

| finding | value |
|---|---|
| training rows (fit) / laundering | 325,015 / 1,672 |
| chosen set-up (3 rolling folds) | warm-up 24 h, weighting=sample, {'num_leaves': 15, 'min_child_samples': 20}, 5 seeds |
| test rows / laundering / weighted prevalence | 205,317 / 1,493 / 0.1139% |
| PR-AUC lgbm_all (95% CI) | 0.5482 (0.4660-0.6070) |
| PR-AUC lgbm_no_format | 0.4254 (0.3626-0.4750) |
| PR-AUC lgbm_no_generator (95% CI) | 0.3674 (0.3109-0.4162) |
| PR-AUC WITH the B16b graph features: all / no_generator | 0.5321 / 0.3651 |
| non-ACH-only model: PR-AUC on non-ACH test rows (prevalence) | 0.0019 vs main model 0.0022 (0.0169%) |
| best F1 % (weighted): lgbm_all / lgbm_no_generator | 59.4 / 43.3 (published HI-Small: GFP+LightGBM 62.86, PNA 56.77; different split) |
| lgbm_all at 0.1%/day: alerts per day, precision / recall | 439, 55.52% / 48.96% |
| lgbm_all at 0.25%/day: precision / recall | 26.31% / 57.94% |
| non-ACH laundering recall: plain budget / 20% of budget reserved | 0.0% / 3.1% (overall recall 65.2% -> 62.8%) |
| PR-AUC logreg_all / rule | 0.2670 / 0.0137 (cycle3_48h) |
| lgbm_all at 0.5%/day: alerts per day | 2,189 |
| lgbm_all at 0.5%/day: precision / recall | 14.83% (13.21%-16.53%) / 65.24% (56.37%-72.81%) |
| lgbm_no_generator at 0.5%/day: precision / recall | 11.64% / 51.17% |
| fixed validation threshold: alert rate on test | 0.77% (target 0.50%) |
| complete schemes caught, lgbm_all / lgbm_no_generator | 106/107 (99.1%) / 100/107 (93.5%) |
| recall explained / unexplained laundering (lgbm_all) | 94.2% / 19.9% |
| no-seed laundering rows left out (all splits) | 0 |
| group split PR-AUC (all / no_generator) | 0.4535 / 0.2659 |

## Test PR-AUC by model

| model | rows | positives | prevalence | pr_auc | pr_auc_lift | roc_auc | pr_auc_95ci |
|---|---|---|---|---|---|---|---|
| lgbm_all | 205317 | 1493 | 0.0011 | 0.5482 | 481.4910 | 0.9855 | 0.4660-0.6070 |
| lgbm_graph_all | 205317 | 1493 | 0.0011 | 0.5321 | 467.3677 | 0.9855 | 0.4546-0.5884 |
| lgbm_no_format | 205317 | 1493 | 0.0011 | 0.4254 | 373.6058 | 0.9722 | 0.3626-0.4750 |
| lgbm_no_generator | 205317 | 1493 | 0.0011 | 0.3674 | 322.6664 | 0.9721 | 0.3109-0.4162 |
| lgbm_graph_no_generator | 205317 | 1493 | 0.0011 | 0.3651 | 320.6675 | 0.9732 | 0.3075-0.4145 |
| logreg_all | 205317 | 1493 | 0.0011 | 0.2670 | 234.5479 | 0.9792 | 0.2134-0.3157 |
| logreg_no_generator | 205317 | 1493 | 0.0011 | 0.0439 | 38.5472 | 0.9500 |  |
| rule | 205317 | 1493 | 0.0011 | 0.0137 | 12.0299 | 0.7329 |  |

## Daily alert budgets (lgbm_all)

| budget | model | alerts | alert_rate | precision | recall | true_positives | lift | alerts_per_day |
|---|---|---|---|---|---|---|---|---|
| 0.001 | lgbm_all | 852 | 0.0010 | 0.5552 | 0.4896 | 731 | 487.5996 | 438.9075 |
| 0.0025 | lgbm_all | 1311 | 0.0025 | 0.2631 | 0.5794 | 865 | 231.0804 | 1095.9024 |
| 0.005 | lgbm_all | 2024 | 0.0050 | 0.1483 | 0.6524 | 974 | 130.2763 | 2188.8330 |
| 0.01 | lgbm_all | 3350 | 0.0100 | 0.0836 | 0.7348 | 1097 | 73.3874 | 4376.2697 |
| 0.02 | lgbm_all | 5926 | 0.0200 | 0.0464 | 0.8158 | 1218 | 40.7634 | 8747.7415 |

## Group split (robustness)

| features | test rows | laundering | prevalence | pr_auc | pr_auc_95ci | time-split pr_auc |
|---|---|---|---|---|---|---|
| all | 99489 | 683 | 0.0011 | 0.4535 | 0.3613-0.5413 | 0.5482 |
| no_generator | 99489 | 683 | 0.0011 | 0.2659 | 0.2045-0.3293 | 0.3674 |

## Before / after the graph and behaviour features (EDA B16b)

| features | PR-AUC without B16b | PR-AUC with B16b | best F1 without | best F1 with |
|---|---|---|---|---|
| all | 0.5482 | 0.5321 | 0.5939 | 0.5764 |
| no_generator | 0.3674 | 0.3651 | 0.4330 | 0.4251 |

## Non-ACH share of the alert budget

| non_ach_share_of_budget | precision | recall | recall_ACH | recall_non_ACH |
|---|---|---|---|---|
| 0.0 | 0.1483 | 0.6524 | 0.7510 | 0.0000 |
| 0.1 | 0.1463 | 0.6403 | 0.7340 | 0.0204 |
| 0.2 | 0.1436 | 0.6276 | 0.7178 | 0.0306 |
| 0.3 | 0.1419 | 0.6196 | 0.7062 | 0.0459 |

Figures: `figures/04_pr_curves_test.png`, `figures/03_flag_lift_train.png`.

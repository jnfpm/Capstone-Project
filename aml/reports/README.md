# reports/ — material for the written report

Written by the notebooks:
- `model_findings.md` — headline numbers, PR-AUC by model, alert budgets, group split (`04_MODELLING.ipynb`, §11)
- `figures/04_pr_curves_test.png` — precision-recall curves on test (`04_MODELLING.ipynb`, §5)
- `figures/03_flag_lift_train.png` — how much each 0/1 feature raises the laundering rate (`03_PREPROCESSING.ipynb`, §4.1)

Put the final report, slides and any hand-made figures here too.

**When quoting results:** PR-AUC always next to the weighted prevalence and its 95% interval; lead with the
`no_generator` model (behaviour a real bank would see) and give `all` as the ceiling inside this synthetic
data; state the limitations printed by `src/02_subsample_split.py` (seed selection used whole-window labels,
weighted precision is high-variance, 10 days of synthetic data).

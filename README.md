# Calibrated Demand Forecasting for Supply Chain Risk Triage

Calibrated uncertainty quantification for weekly demand forecasting — turning a point forecast into a properly calibrated interval that drives an actual inventory decision.

## Methods

- **Method A** — LightGBM Quantile Regression + Conformalized Quantile Regression (CQR)
- **Method B** — Bootstrap Ensemble (10 models) + CQR

Both are calibrated on a held-out fold and compared on coverage, sharpness, and a simulated stocking decision.

## Dataset

[DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle) — weekly demand, top 10 product categories, 2015–2018.

## Results

| Nominal level | Raw QR | Method A | Method B |
|---|---|---|---|
| 90% | 71% | 82% | 81% |
| 95% | 77% | 83% | 84% |

Both calibrated methods cut stockouts by ~97–98% vs. a naive point-forecast policy in the decision simulation.

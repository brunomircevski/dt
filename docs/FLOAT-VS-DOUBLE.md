# Float vs double split scoring: already tried, keep double

Feature values are float32; split scoring (impurities, gains, cut-skipping
bounds, count tables) is double. In November 2026 we tried float scoring on
all datasets, CART and C4.5, on CPU (epyc, serial and 48 threads) and GPU
(RTX 2080 Ti):

- CPU: not faster, often slower (up to +60% on SUSY and HIGGS): float's wider
  rounding margins let the bounds skip fewer cuts.
- GPU: at most 8–30% faster, on a card with weak FP64; about 0 elsewhere.
- Trees change on most datasets, CPU and GPU trees stop matching, and SUSY
  full-depth CART degenerates (0.29 s → 115 s). Accuracy changes are noise
  (±0.09 pp).

The experiment is on the local branch `fp32-experiment`.

## What the compared implementations use

| implementation | split scoring |
|---|---|
| scikit-learn | double (X float32) |
| rpart | double |
| Weka J48 | double |
| YaDT | double |
| cuML RF | float (for float32 input) |
| XGBoost GPU hist | float gain, int64 fixed-point histograms |

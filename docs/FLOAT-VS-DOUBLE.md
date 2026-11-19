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

## Why float is sometimes far slower

Gini is computed as n − Σc²/n, a difference of two numbers near n. In float
that leaves an error of about 1e-7 on every gain, so nearly pure nodes, whose
cuts all gain less than that, can't be ranked. Float's tie threshold (1e-6)
has to exceed this error, so all their cuts count as ties and the first one
wins, which peels off a single row. CART splits even at zero gain, so the
child repeats this, one row at a time.

Example: SUSY, full-depth CART. A depth-10 node has 160,530 rows (124 of one
class); its best cut gains 7.0e-7, a one-row peel 7e-12. Float peels rows off
it for 97,000 levels, each sweeping all 18 features over 11k–160k rows:
116 s instead of 0.29 s. Double (error ~1e-15, ties 1e-12) takes the right
cut and the whole tree is 81 deep.

On a smaller scale, float's wider rounding margins also make the CPU bounds
skip fewer cuts, which is why SUSY and HIGGS are slower even at depth 10.

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

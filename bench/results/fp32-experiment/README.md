# Float vs double split scoring (2026-11-18)

Question: is double precision in the split scoring worth it, or would float be
as good and faster? **Answer: keep double.**

## What was tested

- Feature values were already float32; only the scoring (impurities, gains,
  cut-skipping bounds, count tables) was double. An experiment branch,
  `fp32-experiment` (local, not pushed; worktree `~/dt-fp32`), adds a `real`
  alias that `make FLOAT=1` switches to float on CPU and GPU, building
  `./tree_f32` and `./tree_cpu_f32`. Thresholds and C4.5's 1e-5 gap stay double.
- The double build reproduces main's trees exactly; both builds pass
  `tests/check.sh`.
- Machines: epyc1 48-thread multicore (5 runs), epyc2 serial (3 runs),
  turing RTX 2080 Ti `--gpu` / `--hybrid` (3 runs); double and float runs
  alternated. H100/A100 were busy and not tested.
- All datasets: covertype, SUSY, HIGGS, chess 1M-1B, MNIST8M, ECBDL (15M and
  full), SOREL (4M and full); CART and C4.5, depth 10 and full depth.
  Full per-case numbers: [table.md](table.md).

## Results

- **CPU: float is not faster.** Multicore −11% to +26%, serial up to +60%
  (SUSY CART). Slower on SUSY and HIGGS: float's rounding margins are about
  1e9 times wider, so the bounds skip far fewer cuts. The sweeps are memory-
  and branch-bound, so cheaper arithmetic barely helps.
- **GPU (2080 Ti, FP64 at 1/32 of FP32): at most 8-30% faster**, on SUSY and
  HIGGS only; about 0 elsewhere. Cards with fast FP64 (H100, A100) should gain
  less.
- **Float breaks SUSY full-depth CART:** rounding noise looks like positive
  gain, the tree becomes a 96,994-level chain (1.44M vs 1.19M nodes), and
  training goes from 0.29 s to 115 s (GPU 1.4 s to 27 s).
- **Trees change** on every dataset except covertype depth 10 and chess.
  Test accuracy moves within ±0.09 pp either way: noise.
- **CPU and GPU trees disagree in float** (SUSY, HIGGS, covertype); in double
  they always match.

## What the compared implementations use

| implementation | split math |
|---|---|
| scikit-learn | double (X float32) |
| rpart | double |
| Weka J48 | double |
| YaDT | double (`gain()`/`info()` take `vector<double>`) |
| cuML RF | float (`DataT` for float32 input); counts int64 |
| XGBoost GPU hist | float gain; exact int64 fixed-point histograms |

Every exact CPU tool uses double, like ./tree. The GPU libraries use float but
score only 128-256 histogram bins per feature, not every cut.

## Bug found

`skippableSteps()` in `src/build/cpu_sweep.h` converted a possibly negative
step count to `uint32_t` (undefined behaviour). In double the value is always
above −1, so it truncated to 0 and did no harm; in float it wrapped and skipped
past the best cuts. Fixed on main by returning 0 when `steps <= 0`; no double
tree changes.

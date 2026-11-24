# Section 5 (Results): layout plan by `opus`

The plan rests on one rule: **every number is printed in exactly one place.**
- Fig. 5 is the only place for training times at depth ≤ 10, both ours and the libraries'.
- One table holds the attributes of each tree: nodes, test accuracy, RAM and VRAM.
- The depth figure is the only place for deep trees.
- The scalability figure is the only place for thread and sample scaling.
- The energy table and Fig. 6 split the energy numbers between them.

The two current time tables (`tab:time-cart`, `tab:time-c45`) duplicate Fig. 5 number for number, so their *content* goes into Fig. 5. Their *style* lives on in the new tree and memory table (T1) and in the energy table (T2).

Budget: 4 figures + 2 tables ≈ 2.6 pages of floats, plus ≈ 1.6 pages of text, for **≈ 4.2 pages** in all.

---

## 1. Subsections

### 5.1 Training Time of the Five Implementations (Fig. 5 extended, F1)
- **Must show:**
  - the ladder Serial Simple → Serial Optimized → Multicore → GPU-Based / Hybrid on all 7 dataset rows, CART and C4.5;
  - the libraries next to ours on HIGGS and MNIST8M;
  - the second node of each kind (dell, A100) as markers, so that hardware needs no subsection or table of its own.
- **Claims, with numbers checked in `bench/cluster-*/*/summary.csv`** (CART; C4.5 differs by <10 %):
  - **Serial Opt ÷ Simple:** 16.5× (MNIST8M) to 30× (HIGGS).
  - **Multicore ÷ Serial Opt:** 14.6× (SUSY) to 40× (SOREL full: 608 ÷ 15.1).
  - **GPU-Based (H100) ÷ Multicore (epyc):** 1.16× (SUSY) to 1.69× (SOREL 4M).
  - **Hybrid at depth 10:** 3–70 % slower than GPU-Based. The hand-off buffer set-up costs about 0.1 s, so on SUSY and HIGGS Hybrid even loses to the CPU (0.262 vs 0.179 s). This sets up 5.3.
  - **Libraries, one thread vs one thread** (Serial Opt): scikit-learn 27× / 8.8×, J48 103× / 147×, YaDT 12× / 2.3× (HIGGS / MNIST8M).
  - **Libraries, GPU** (vs GPU-Based): cuML 2.6× / 2.4×, XGBoost 2.3× / 3.3× (`cluster-cpu-vs-others`, `cluster-gpu-vs-others/one-tree`).
  - **Hardware:**
    - dell Multicore is 1.4× (SUSY) to 2.8× (SOREL C4.5) slower than epyc. Bandwidth: 8 vs 12 channels.
    - The A100 is 1.10–1.52× slower than the H100, except SUSY, where it is 8 % faster: set-up dominates there.
    - On ECBDL'14 15M the A100 server is no faster than epyc's CPUs (4.30 vs 4.47 s).
  - The intro's ratios live here, so they are not repeated in text.

### 5.2 Tree Quality and Memory (Table T1)
- **Must show:**
  - all five implementations grow one tree per dataset (nodes, test accuracy);
  - the libraries' trees: J48 923 vs 935 nodes; YaDT's different C4.5 on HIGGS; rpart's 1,181-node tree; binned GPU trees within 0.05 points at depth 10;
  - the binned trees falling behind at depth 30 on HIGGS: 68.77 % exact vs 68.19 % for cuML and XGBoost, with 2.16 M vs 2.61 M / 2.34 M nodes;
  - **the price of exactness, in memory:**
    - presorted columns cost about 3× the data in host RAM: MNIST8M 71–74 GiB for 25.4 GB of data, SOREL full 345 GiB;
    - they cost about 2× in VRAM, which is why the full ECBDL'14 and SOREL do not fit;
    - GPU runs keep host RAM at about 1× the data, while scikit-learn needs 24 GiB and J48 82 GiB on MNIST8M;
  - pruned accuracy, in one sentence (standing decision), from a **NEW RUN** (§4).
- This is the only subsection where accuracy appears. Fig. 5 carries no accuracy.

### 5.3 Deep Trees: Where Hybrid Beats GPU-Based (F2, replaces `test-depth`)
- **Must show:** time against depth limit (5 … 30, none) for Hybrid and GPU-Based on H100 and A100, for HIGGS, MNIST8M and SOREL 4M, in CART and C4.5. Then *why*: the MNIST8M "who is working" profile.
- **Claims (`hybrid-vs-gpu-only/BENCHMARK.md`, hopper):**
  - GPU-Based wins to depth 10 (Hybrid 0.63–0.97×).
  - Break-even comes at depth 15 on MNIST8M and SOREL, and at 25 on HIGGS.
  - With no limit, Hybrid wins: HIGGS 1.28× / 1.76×, MNIST8M 1.52× / 1.61×, SOREL 4M 1.18× / 1.08×. On ampere: 1.75×, 1.41–1.46×, 1.07–1.16×.
  - On MNIST8M with no limit, the GPU and CPU overlap for 61 % (CART) and 74 % (C4.5) of Hybrid's time, while the GPU-Based "host" share is 14–18 % (`profile/`, `chart-profile-hopper-mnist8m.png`).
  - GPU libraries at depth 30 on HIGGS: cuML 4.18 s and XGBoost 23.8 s, against 0.99 s for GPU-Based and 0.82 s for Hybrid. This is the deep-tree library point.

### 5.4 Scalability in Threads and Samples (F3, merges `test-scaling` and the chess profiling)
- **Must show:**
  - thread scaling 1–48 on epyc and dell: HIGGS 22.8× / 19.1× CART, MNIST8M 34.9× / 25.9× (`cluster-cpu-scaling/*/scaling.csv`);
  - **chess, cost per sample from 10⁷ to 2·10⁹ rows, split into the five profiled steps**, for epyc Multicore, dell Multicore, H100 GPU-Based and Hybrid.
- **Claims:**
  - **GPU, per sample:** 13 ns/sample at 10 M (fixed set-up) falls to 0.76 ns at 2 B (1.52 s).
  - **epyc, per sample:** a flat ~1.7 ns/sample (3.44 s at 2 B); dell ~4.5 ns.
  - **Crossover:** GPU beats epyc from about 100 M rows on.
  - **CPU steps:** partition is 45–51 % on epyc.
  - **GPU steps:** CPU↔GPU transfer grows to 27–31 % on the GPU.
  - **Real datasets, text only:** partition 44 % (HIGGS) and 69 % (ECBDL'14 15M) on epyc; the GPU is busy 74–80 % of the time on ECBDL (`cluster-profiling/BENCHMARK.md`).

### 5.5 Energy (Table T2 + Fig. 6)
- Keep the current text's argument (energy ∝ time; the H100 server is cheapest; Hybrid saves 4–10.5 % at full depth).
- T2 gains the MNIST8M no-depth block, so that every *t / P / E* number lives in T2. Fig. 6 then shows only what the table cannot: the idle / CPU / GPU split.

*(No separate "Profiling" or "Performance across hardware" subsections: hardware sits in F1, profiling in F2 and F3.)*

---

## 2. Figures and tables

### F1 = Fig. 5 `fig:test-bars`, extended (full width, 2 rows CART / C4.5, ≈ 3.2 in ≈ 0.55 page)
- **Source:** `charts.py:bars()` with three changes.
  1. **Libraries in the HIGGS and MNIST8M groups.**
     - Bars: hollow, hatched; the colours are already in `STYLE`. CART row: scikit-learn, rpart, cuML, XGBoost. C4.5 row: J48, YaDT, YaDT 48t.
     - Those two groups get wider slots (variable group width).
     - "failed" text for rpart on MNIST8M.
     - Sources: `cpu_other()` and `gpu_other()` in `charts.py`; a YaDT key needs adding to `STYLE`.
  2. **Second-node markers.**
     - A short black tick across each Serial Opt and Multicore bar at the dell value; a tick on each GPU-Based and Hybrid bar at the A100 value.
     - Legend entry: "— dell / A100".
     - Sources: `cluster-*/{serial,cpu}-dell-*`, `{gpuonly,cuda}-ampere-*`.
     - Serial Simple gets no dell tick (within 2–9 %, clutter).
  3. **Step ratios** as small grey "×n" labels between adjacent bar clusters? **No:** that repeats the ladder. Keep the vertical value labels; F1 is the one place these values appear.
- **Drops:** `tab:time-cart`, `tab:time-c45`, `test-ladder` (its step ranges go into the text), `test-heatmap`, `test-dots`.
- **Caption:** keep the current one; add "libraries hollow; ticks: dell (CPU) / A100 (GPU)" and the run counts (Simple full ECBDL/SOREL 3, J48 MNIST8M 3).

### T1 `tab:trees-memory`, Trees and memory (≈ 0.4 page, current booktabs style, `\setlength\tabcolsep{4pt}`)
- **Block "This work"** (identical trees in all five), 7 rows: SUSY, HIGGS, ECBDL'14, its 15M, MNIST8M, SOREL-20M, its 4M.

  | col | content | source |
  |---|---|---|
  | CART nodes, acc. (%) | depth ≤ 10 | `cluster-<d>/*/summary.csv` `nodes`, `test_accuracy` |
  | C4.5 nodes, acc. (%) | depth ≤ 10 | same |
  | RAM (GB): Simple / Opt / Multicore | peak RSS, epyc | `peak_rss` |
  | GPU-Based RAM / VRAM (GB) | H100 | `peak_rss`, `gpu_peak` |

  Hybrid is not a column: it is within ±0.2 GB of GPU-Based in RAM and ±1 GB in VRAM (one caption sentence).
- **Block "Existing implementations"**, HIGGS | MNIST8M: rows scikit-learn, rpart, J48, YaDT (1 / 48 thr.), cuML, XGBoost. Columns: nodes, acc., RAM, VRAM.
  - Plus one row pair **"HIGGS, depth ≤ 30"**: ./tree exact | cuML | XGBoost, with nodes and accuracy (from `cluster-gpu-vs-others/summary.csv`). The times stay in F2.
- **ECBDL'14 accuracy:** 98.3 % is the majority-class rate. Print it with a dagger and a footnote, or replace it with balanced accuracy (not recorded; see §4).
- **Units:** bench is in GiB, Table 1 (datasets) in GB. Convert everything to GB.

### F2 `fig:depth`, Hybrid vs GPU-Based by depth (full width, ≈ 2.3 in ≈ 0.45 page; replaces `test-depth`)
- **Left: 2×3 small multiples** (rows CART / C4.5; columns HIGGS, MNIST8M, SOREL 4M).
  - x = depth limit (5, 10, 15, 20, 25, 30, ∞), categorical.
  - y = time, log, shared per column.
  - Lines: Hybrid solid green, GPU-Based grey; H100 filled markers, A100 hollow and dashed (four lines).
  - Shading where Hybrid wins on the H100.
  - Label at ∞: the H100 speed-up "×1.52".
  - Top axis: tree nodes at each depth (CART row only, from `nodes`), which shows *why*: 2 k → 302 k nodes.
  - **HIGGS CART panel only:** two hollow markers at depth 30 for cuML (4.18 s) and XGBoost (23.8 s, an off-scale arrow with its value).
  - Source: `hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-*/summary.csv`, `cluster-gpu-vs-others/summary.csv` (`depth_limit` 30).
- **Right column (spans both rows, ≈ 1.1 in wide): "who is working",** MNIST8M CART on the H100.
  - Paired 100 % bars for Hybrid / GPU-Based at depth 10, 20 and ∞.
  - Segments: set-up, GPU alone, GPU+CPU, CPU alone, transfer, host.
  - Total seconds on top.
  - Source: `hybrid-vs-gpu-only/profile/{cuda,gpuonly}-hopper-mnist8m` (`hybrid_profile_chart.py` logic).
- C4.5 profile is text only (74 % overlap).
- The current `test-depth` bars with "1.2×" labels carry 42 numbers on 6 panels for H100 only. This figure adds A100, the libraries and the mechanism at the same height.

### F3 `fig:scaling`, Scalability (full width, ≈ 3.0 in ≈ 0.5 page; merges `test-scaling` + `cluster-profiling/chart-chess.png`)
- **Row (a), thread scaling, 2 panels:**
  - The current `scaling()` as is: epyc, dell; HIGGS and MNIST8M; CART solid, C4.5 dashed.
  - Ideal line, 2nd-socket line, 48-thread speed-up and efficiency labels.
  - Height reduced to 1.25 in.
- **Row (b), chess cost per sample, 4 small multiples** (epyc Multicore, dell Multicore, H100 GPU-Based, H100 Hybrid):
  - x = training rows, log, 10⁷ … 2·10⁹ (10M, 50M, 100M, 500M, 1B, 2B).
  - y = ns per sample (training time ÷ rows), log, shared.
  - Encoding: stacked areas (or stacked bars at the 6 sizes) of the five steps (presort, split search, partition, transfer, other, in the `profile_share_chart.STEPS` colours); total = unprofiled time from `cluster-profiling/plain/*` and shares from the profiled runs (`cluster-profiling/<mode>-<node>-chess_*`).
  - The epyc total is overlaid as a dotted line on the GPU panels to show the crossover.
  - CART only. C4.5 is +5–15 %, one caption sentence.
- **Drops:** `test-profile` (its chess-1B bars become row (b); its HIGGS and ECBDL shares go to text).
- **Optional extension:** 1M and Serial Optimized from `cluster-chess` (1M–1B). **Not** recommended: a second measurement of the same points (0.741 vs 0.713 s at 1B GPU-Based); use `plain/` only.

### T2 `tab:energy` (≈ 0.25 page) + Fig. 6 `fig:energy` (≈ 0.4 page)
- **T2:**
  - Keep the current block (ECBDL'14 15M, 5 implementations).
  - Add a block "MNIST8M, no depth limit, hopper": Multicore, GPU-Based, Hybrid, each with *t*, *P*, *E* for CART and C4.5 (`cluster-power/mnist8m-nodepth/summary.csv`).
  - Add a column *E*−idle (kJ), which the text quotes now (0.89 vs 2.27 kJ).
- **Fig. 6:** keep both panels (`test-energy*.pdf`), but print no *t / P / E* value labels that T2 holds. Label only the idle share and the CPU and GPU components. Its job is the decomposition.

### Moved out of Section 5
- **`diagram-small-nodes` (Fig. 8):** to Section 3.5 (Hybrid) as motivation, if space allows; otherwise cut. It is a schematic and a level histogram, not a result.
- **`diagram-task-parallel`:** Section 3.3.

---

## 3. Deliberately left out, or moved to Section 6 / appendix
- **Hand-off sweep** (`hybrid-vs-gpu-only/sweep`):
  - The paper says *m* is tuned (best case). One sentence in 5.3: the default rule is within 2 % of the tuned *m*, and 6–12 % separates the old default from the best.
  - The sensitivity discussion goes to Section 6.
- **GPU library phase timelines** (`cluster-gpu-vs-others/chart.png`): text only, two sentences in 5.1. The build is 4.5 s vs 1.43 s, XGBoost's binning 2.35 s, and the presort is hidden in the upload. Section 4 already says the phases are a coarse guide, so a figure would over-weight them.
- **Ladder, heatmap, dots, pareto:** subsumed by F1 + T1.
- **One-socket runs, `ampere-dell-2026-11-16` power history, dell1 runs, chess 1M, Serial / Simple on chess:** not needed for any claim.
- **Medians, min–max whiskers:** means only (standing decision). Bars and lines carry no whiskers.
- **To Section 6:**
  - why depth 10 underfits (C4.5 63.7 % on HIGGS vs 70.1 % for YaDT's different tree);
  - the memory trade-off, exact vs binned;
  - when to choose GPU over CPU (set-up dominates below ~10⁸ samples × features).

---

## 4. Data gaps and inconsistencies

**NEW RUN NEEDED**
1. **Pruned accuracy** (req. 5): no pruned runs exist in `bench/` (only `*_depth10*` unpruned and the depth sweep).
   - Run `run.py <d> --protocols cart_alpha_nodepth,c45 --impls tree --threads all --reps 1`, Multicore on epyc. Datasets: SUSY, HIGGS, MNIST8M, SOREL 4M, ECBDL 15M.
   - Untimed, minutes each. Harness `bench/tools/cluster/submit.sh`.
   - Optionally the same for scikit-learn `ccp_alpha` / J48 `-C 0.25` on HIGGS, for a pruned library comparison. YaDT `-ebpg` is in `benchlib` `c45`.
2. **GPU libraries on ampere:** the decision of 2026-11-19 says "everything on both GPUs". Only `vs-hopper-*` exists. Run `final.sh -k gpulibs` on ampere, or drop the A100 tick for the libraries in F1 (my preference, if time is short).
3. **(Optional) balanced accuracy for ECBDL'14:** it would need a scoring pass over saved trees; else mark it majority-class.
4. **(Optional) test accuracy in the depth sweep:** `hybrid-vs-gpu-only/*/summary.csv` has an empty `test_accuracy`. With it, F2's top axis could show accuracy vs depth: the underfit at depth 10, overfitting at ∞. This is a cheap rerun of the same harness with test scoring on.

**Inconsistencies found**
- **HIGGS GPU-Based CART:** 0.221 s (`cluster-higgs`) vs 0.213 s (`gpu-vs-others`, job 789), the same build. Use one source per cell: F1 reads `cluster-*` everywhere; library ratios in the text read `gpu-vs-others`' own ./tree row (same run.py).
- **HIGGS Multicore C4.5:** 0.366 s (`cluster-higgs`) vs 0.387 s (`cpu-vs-others`). The same rule applies.
- **HIGGS depth 30 Hybrid:**
  - 0.820 s in `gpu-vs-others` uses the *default* *m* = 512.
  - 0.847 s in `hybrid-vs-gpu-only` uses the "tuned" no-depth *m* = 256.
  - The tuned run is slower, and the `final.sh` gap from 11-19 is still open. F2 should take its depth-30 Hybrid from `hybrid-vs-gpu-only` only, and quote the library comparison as "vs GPU-Based".
- **Sweep coverage:** `test-depth`'s caption says *m* comes from "the depth-10 value up to depth 10". The sweep in `hybrid-vs-gpu-only/BENCHMARK.md` ran only at no limit, but `bench/tools/handoff/handoff_rows.json` holds `depth10` values. Say where those came from (a separate depth-10 sweep is not documented in that BENCHMARK.md).
- **Abstract / intro "scales up to 40× across 48 cores":** this is Multicore ÷ Serial Optimized (SOREL). Same-code thread scaling is at most 34.9× (MNIST8M, `cluster-cpu-scaling`). Word it as "Multicore is up to 40× faster than Serial Optimized", or cite 35×.
- **ECBDL'14 test accuracy:** 98.30–98.32 % is the majority-class rate. Accuracy on ECBDL carries no information.
- **Units:** GiB (bench) vs GB (paper Table 1, "80 GB" GPUs). Pick GB throughout.
- **SUSY:** the A100 GPU-Based (0.142 s) beats the H100 (0.154 s). This is real (set-up dominated), not an error, but the text should not claim "H100 always faster".

---

## 5. Where I expect disagreement
1. **Dropping both time tables** and moving all depth-10 times, the libraries' included, into Fig. 5.
   - The author likes the tables. But Fig. 5 is mandatory and already prints every value, and requirement 7 forbids repeating them.
   - The table style survives in T1 and T2.
   - Fallback if the author insists on a time table: keep the table and strip Fig. 5's value labels. Never both.
2. **No separate hardware or profiling subsections.**
   - dell and A100 fit as ticks on F1.
   - Profiling splits into what it explains: the chess cost per sample in F3, and "who is working" in F2.
   - A heatmap or a hardware table would re-print epyc and H100 numbers.
3. **Energy:** T2 owns all the numbers, Fig. 6 only the decomposition.
   - Others may keep Fig. 6's labels and drop T2.
   - I keep T2 because it carries CART and C4.5 and the MNIST8M block in a quarter page; the figure cannot do that legibly.
4. **GPU library phase timelines as text, not a figure.** The totals are the comparison (Section 4 says so). The deep-tree library points (4.2 s and 23.8 s at depth 30) are more persuasive and fit into F2 for free.
5. **Chess as ns/sample stacked by step** instead of the bench's 100 % share bars.
   - Shares hide the fact that the GPU's per-sample cost keeps falling while the CPU's stays flat.
   - Absolute ns/sample shows both the crossover (~10⁸) and the step that dominates at each size.

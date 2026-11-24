# Round 2: Cross-Critique and Revised Plan — Gemini 4 Neon

Target budget: **~5.8 LNCS pages** (5 tables ≈ 1.44 p, 6 figures ≈ 2.53 p, prose ≈ 1.85 p; down from 6.7 p in Round 1), inside the 5–6 page target with zero duplicated numbers across tables and figures.

---

## 1. Resolution of Disagreements

1. **Time tables vs Fig. 5 (`test-bars`):** `NEW-COMPROMISE` — Keep both existing-style time tables (`tab:time-cart` with A100 cuML/XGBoost rows `[NEW RUN]`, `tab:time-c45`) as the sole source of exact depth-10 seconds, and **strip all per-bar numeric labels from Fig. 5** while upgrading Fig. 5 with hollow/hatched library bars (`opus`), `dell`/`A100` tick markers (`opus`), and subtle adjacent-step speed-up brackets (`gptsol`, retiring `test-ladder`).
   - *Reason:* Honors the author's binding request for tables in the existing style and mandatory Fig. 5 while eliminating the number-for-number duplication that flawed my Round 1 plan.
2. **Performance across hardware (`epyc`/`dell`, H100/A100):** `AGREE-WITH opus` — Drop my separate §5.4 and hardware heatmap; encode `dell` and `A100` as black tick marks directly on Fig. 5 (extending to A100 cuML/XGBoost once run), keep A100 vs H100 across depths in `fig:depth`, and keep `epyc` vs `dell` in `fig:scaling` and `fig:chess-profile`.
   - *Reason:* Four of my heatmap's six columns duplicated Table 1; Fig. 5 ticks show both CART and C4.5 across all 7 datasets and both CPU/GPU pairs at zero extra page cost.
3. **GPU-library phase timeline:** `KEEP-MINE` (aligned with `gptsol`) — Keep a compact 3-row H100 Gantt figure (`fig:gpu-timeline`, ~0.44 p) in §5.2 for HIGGS d10, MNIST8M d10, and HIGGS d30 (broken x-axis), annotating co-scheduled ratios on d10 (avoiding Table 1 duplication) and exact times/nodes/acc on d30.
   - *Reason:* Exact GPU presorting beating binned cuML/XGBoost by 2.4–29× is a flagship counter-intuitive result; seeing presort hide inside upload while XGBoost bins (2.35 s) and both libraries stall in build (4.5 vs 1.43 s) is far more persuasive than two prose sentences.
4. **RAM / VRAM placement:** `AGREE-WITH opus` — Consolidate peak host RAM and peak VRAM for all 7 datasets (our backends) and all HIGGS/MNIST8M CPU/GPU libraries into one booktabs table `tab:trees-memory` in §5.2, dropping my separate memory bar panel.
   - *Reason:* Unifies the exactness-vs-memory trade-off in one table in the author's style and recovers 0.37 p while still exposing why full ECBDL'14/SOREL-20M exceed 80 GiB VRAM.
5. **Chess cost per sample, step shares, sizes, and thread scaling:** `NEW-COMPROMISE` — Keep `fig:scaling` (`test-scaling.pdf`, 0.27 p) separate from `fig:chess-profile` (0.48 p); in `fig:chess-profile`, plot **(a)** 10M–2B `ns/sample` on a **log-y line chart** (never stacked on a log axis, which distorts segment proportions in `opus`'s F3!) and **(b)** 100% linear stacked step-share bars across Chess 10M–2B `|` HIGGS, ECBDL'14 15M (`▼` GPU-busy), reading exclusively from `cluster-profiling/` (citing 1M/A100/Serial from `cluster-chess` in prose).
   - *Reason:* Keeps log `ns/sample` visually honest, preserves real-dataset profiling (69% Partition / 77% GPU-busy), avoids splicing `cluster-chess` 1B (0.741 vs 0.713 s), and avoids an oversized 0.82 p mega-float.
6. **Hybrid-wins figure:** `AGREE-WITH opus` — Adopt `opus`'s 2×3 log-time line grid (HIGGS, MNIST8M, SOREL 4M × CART, C4.5; H100 filled + A100 hollow dashed; Hybrid-win shading; top node-count axis; HIGGS CART d30 cuML/XGBoost markers) paired with a right-column MNIST8M CART "who is working" 100% stacked-bar panel (Hybrid vs GPU-Based at d10, d20, ∞, annotating total s and CPU thread-s; C4.5's 74% overlap in caption/prose).
   - *Reason:* Log-time lines beat my grouped bars (showing A100 across all depths) and beat `gptsol`'s ratio plot (showing A100 48T beating H100 32T on HIGGS ∞, actual runtimes, and d30 library gaps), while paired bars show GPU-Based's 14–18% host overhead collapsing to 1%.
7. **Energy (`tab:energy` vs Fig. 6):** `AGREE-WITH opus` — Keep `tab:energy` expanded with the 3-row MNIST8M full-depth block and an `E - idle` column (carrying every exact CART & C4.5 `t / P / E` value), and strip redundant `t / P / E` bar-end labels from Fig. 6 so Fig. 6 shows only the idle/dynamic and CPU/GPU split.
   - *Reason:* Mirrors Point 1 cleanly, retains all CART numbers that C4.5-only Fig. 6 omits, and honors the author's preference for `tab:energy` without repeating numbers.
8. **Accuracy (pruned table vs exact-vs-binned):** `NEW-COMPROMISE` — Keep exact-vs-binned d10 & HIGGS d30 fidelity in `tab:trees-memory`, and add a compact `tab:pruned` (`NEW RUN NEEDED`, ~0.22 p) comparing unpruned full-depth vs pruned (`ccp_alpha=1e-5`, `CF=0.25`) across SUSY, HIGGS, MNIST8M, SOREL 4M, and ECBDL'14 15M (flagging its 98.3% majority-class rate) plus 3 HIGGS pruned CPU-library rows (sklearn, J48, YaDT).
   - *Reason:* Fitting pruned columns into `tab:trees-memory` overflows LNCS width, while one sentence (`opus`) under-delivers Requirement 5; a compact 0.22 p table documents C4.5 depth-10 underfitting and pruned library parity.
9. **Number and order of subsections:** `AGREE-WITH gptsol` — Converge on **5 subsections** (dropping my old §5.4): §5.1 Overall Training Time & Hardware → §5.2 Libraries, Accuracy & Memory → §5.3 Deep Trees (Hybrid) → §5.4 Scalability & Profiling → §5.5 Energy.
   - *Reason:* Keeps §5.1 focused on our ladder + hardware, groups all external comparisons/fidelity/memory cleanly in §5.2, and puts Hybrid's deep-tree win center stage.
10. **Duplicated measurements rule:** `AGREE-WITH opus` (refining mine/`gptsol`) — Single-source per visual/table, zero cross-harness splicing: `cluster-*` feeds Tables 1–2 ("This work") and Fig. 5; `cluster-{cpu,gpu}-vs-others` feeds library entries and **all** library speed-up ratios (using each job's co-scheduled `./tree` row, e.g., 0.213 s / 0.387 s, noting 1–6% cross-job drift once); `hybrid-vs-gpu-only` feeds `fig:depth` (d30 Hybrid `0.847 s`, `m=256`) while `gpu-vs-others` feeds `fig:gpu-timeline` (d30 Hybrid `0.821 s`, `m=512`, stating `m`); `cluster-profiling/` exclusively feeds `fig:chess-profile`.
    - *Reason:* Guarantees every reported ratio and curve comes from one consistent measurement protocol.
11. **Extra conflict — Memory units (`GiB` vs `GB`):** `KEEP-MINE` (agrees with `gptsol`) — Standardize Section 5 on **GiB** (binary $2^{30}$) and add one clarifying note in §4.1/§4.2 that "80 GB" GPUs provide 80 GiB (85.9 GB).
    - *Reason:* Converting SOREL 4M's `75.9 GiB` VRAM to decimal `81.5 GB` creates an apparent contradiction next to "80 GB GPU".

---

## 2. Revised Complete Layout (~5.8 LNCS Pages)

*Existing Section 3 moves retained:* `diagram-task-parallel.pdf` → §3.3, `diagram-small-nodes.pdf` → §3.5. *Dropped:* `test-ladder`, `test-heatmap`, `test-dots`, `test-pareto`, `test-profile`, and standalone `$m$`-sweep plots (summarized in 2 sentences in §5.3).

### 5.1 Overall Training Time and Hardware Comparison (~1.42 p: 1.02 p floats + 0.40 p prose)
- **Must show:** Canonical depth-10 training times for all 5 backends on all 7 dataset rows (`epyc`/`hopper`), external libraries on HIGGS & MNIST8M (incl. A100 GPU libraries), optimization progression, and `epyc` vs `dell` + H100 vs A100 hardware comparison.
- **Table 1 (`tab:time-cart`, ~0.29 p):** Existing style (tighten `\tabcolsep` to fix 6.1 pt overfull). Top: 7 dataset rows × 5 backends (`epyc`/`hopper`). Bottom: HIGGS & MNIST8M × scikit-learn, rpart (`--` on MNIST8M), cuML (H100), XGBoost (H100), plus 2 rows for **cuML & XGBoost on A100 [NEW RUN NEEDED]**.
  - *Sources:* `bench/cluster-{susy,higgs,mnist8m,ecbdl,sorel}/*/summary.csv`, `bench/cluster-{cpu,gpu}-vs-others/` (+ `final.sh -k gpulibs` on `ampere`).
- **Table 2 (`tab:time-c45`, ~0.27 p):** Existing style. Top: 7 dataset rows × 5 backends. Bottom: HIGGS & MNIST8M × Weka J48, YaDT (1T, 48T). *Sources:* same as Table 1.
- **Fig. 5 (`fig:test-bars`, redesigned, ~0.46 p):** 2 rows (CART top, C4.5 bottom), log-y grouped bars across all 7 datasets, **no per-bar numeric labels**. Adds: (i) hollow/hatched library bars in wider HIGGS & MNIST8M slots (`failed` for rpart), (ii) short black `—` ticks for `dell` on `Serial Opt`/`Multicore` and `A100` on `GPU-Based`/`Hybrid`/cuML/XGBoost, (iii) subtle step speed-up range brackets (`16–35×`, `14–40×`, `1.1–1.7×`) retiring `test-ladder`.
  - *Sources:* `paper/tools/charts.py:bars()` extended with `{serial,cpu}-dell-*`, `{gpuonly,cuda}-ampere-*`, `cpu_other()`, `gpu_other()` (hopper + ampere `[NEW RUN]`).

### 5.2 Comparison with Existing Libraries, Tree Quality, and Memory (~1.46 p: 1.06 p floats + 0.40 p prose)
- **Must show:** Co-scheduled CPU & GPU library speed-ups (incl. A100 ratios once run), why exact GPU presorting beats binned cuML/XGBoost via phases, exact-vs-binned fidelity (d10 and HIGGS d30), pruned full-depth accuracy, and peak host RAM / VRAM across all 7 datasets and all libraries.
- **Fig. 6 (`fig:gpu-timeline`, ~0.44 p):** 3-row H100 Gantt chart: (a) HIGGS d10, (b) MNIST8M d10, (c) HIGGS d30 (broken x-axis `//` for cuML 4.18 s, XGBoost 23.84 s). Four lanes (`GPU-Based`, `Hybrid`, cuML, XGBoost) showing overlapping `setup`, `upload`, `prepare` (presort vs binning), `GPU build`, `CPU build`. Right labels: co-scheduled relative ratio (`1.0×`, `2.7×`, `2.4×`) on d10 (avoiding Table 1 duplication); exact total s + ratio on d30. A100 ratios noted in prose/caption `[NEW RUN]`.
  - *Sources:* `bench/cluster-gpu-vs-others/summary.csv`, `one-tree/vs-hopper-{higgs,mnist8m,higgs-depth30}/results.jsonl`.
- **Table 3 (`tab:trees-memory`, ~0.40 p):** Unified booktabs table (units: **GiB**).
  - *Top ("This work", 7 rows):* CART d10 (nodes, acc %), C4.5 d10 (nodes, acc %), Peak RAM (`Simple` / `Multicore` / `GPU-Based`), Peak VRAM (`GPU-Based`). Caption notes `Serial Opt` ≈ `Multicore` RAM (within 1–3 GiB) and `Hybrid` ≈ `GPU-Based` (±0.2 GiB RAM, ±1 GiB VRAM); ECBDL'14 acc marked `†` (98.3% majority class).
  - *Bottom ("Existing implementations", HIGGS & MNIST8M):* sklearn, rpart, J48, YaDT (1T, 48T), cuML, XGBoost + **HIGGS d30** (`./tree` exact, cuML, XGBoost) showing nodes, test acc (%), Peak RAM, Peak VRAM.
  - *Sources:* `bench/cluster-*/*/summary.csv`, `bench/cluster-{cpu,gpu}-vs-others/`.
- **Table 4 (`tab:pruned`, ~0.22 p, NEW RUN NEEDED):** Compact booktabs table, untimed (`--reps 1`). Top: SUSY, HIGGS, MNIST8M, SOREL 4M, ECBDL'14 15M (`†` majority class; optional balanced acc) × CART (`unpruned ∞` vs `ccp_alpha=1e-5`: nodes, test acc) and C4.5 (`unpruned ∞` vs `CF=0.25`: nodes, test acc). Bottom: HIGGS pruned CPU libraries (scikit-learn `ccp_alpha`, J48 `-C 0.25`, YaDT `-ebpg`).
  - *Harness:* `bench/tools/run.py --protocols cart_full_nodepth,cart_alpha_nodepth,c45_unpruned,c45 --reps 1 --test-accuracy` on `epyc` (`submit.sh`).

### 5.3 Deep Trees: When Hybrid Outperforms GPU-Based (~0.81 p: 0.46 p float + 0.35 p prose)
- **Must show:** Depth crossover (5→30, ∞) across HIGGS, MNIST8M, SOREL 4M on both H100 and A100, why Hybrid overtakes GPU-Based (GPU+CPU concurrency, host overhead collapse), deep-tree GPU library gap at d30, tree-shape limit (thin SOREL trees), and 2-sentence `$m$`-default summary.
- **Fig. 7 (`fig:depth`, ~0.46 p, replaces `test-depth.pdf`):** Two-part dense figure:
  - *Left (2×3 small multiples, CART/C4.5 × HIGGS/MNIST8M/SOREL 4M):* x = depth limit (5..30, ∞), y = time (log, shared per column); 4 lines (Hybrid green / GPU-Based grey × H100 filled solid / A100 hollow dashed), shaded Hybrid-win region (H100), H100 ∞ speed-up label (`×1.52`), top axis showing CART tree node counts. HIGGS CART panel includes hollow d30 markers for cuML (4.18 s) and XGBoost (23.8 s, off-scale arrow; A100 markers optional post-NEW RUN).
  - *Right column (spans both rows):* Paired 100% stacked "who is working" bars for Hybrid vs GPU-Based on MNIST8M CART (H100) at depths 10, 20, ∞ (`setup`, `GPU alone`, `GPU+CPU`, `CPU alone`, `transfer`, `host`), total seconds on top, CPU thread-s annotated (C4.5 74% overlap in prose/caption).
  - *Sources:* `bench/hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-*/summary.csv`, `profile/*-hopper-mnist8m/`, `cluster-gpu-vs-others/summary.csv` (d30).

### 5.4 Scalability and Execution Profiling (~1.13 p: 0.75 p floats + 0.38 p prose)
- **Must show:** Thread scaling 1→48T on `epyc` vs `dell`, Chess $3\times 3$ cost per sample ($10^7 \to 2\cdot 10^9$ rows) showing CPU/GPU crossover (~100M) and linear asymptote (`0.76 ns/sample` at 2B), and 5-step execution profile across Chess 10M→2B plus real datasets (HIGGS, ECBDL'14 15M).
- **Fig. 8 (`fig:scaling`, ~0.27 p):** Keep `test-scaling.pdf` (2 panels: `epyc`, `dell`; 1–48T speedup for HIGGS/MNIST8M, CART solid / C4.5 dashed, ideal line, 24T socket line + subtle 12T `dell` sub-NUMA tick, 48T speedup/efficiency labels). *Sources:* `bench/cluster-cpu-scaling/cpu-{epyc,dell}-{higgs,mnist8m}/scaling.csv`.
- **Fig. 9 (`fig:chess-profile`, ~0.48 p, replaces `test-profile.pdf`):** Dense 2-row figure from `cluster-profiling/` exclusively:
  - *(a) Cost per sample (10M→2B):* log-x (rows) vs log-y (**ns/sample**), lines for `epyc 48T`, `dell 48T`, `H100 GPU`, `H100 Hybrid` (CART solid, C4.5 dashed), setup-dominated vs asymptotic regions, crossover (~100M) and 2B total-s (`1.52 s`, `3.44 s`, `9.02 s`) annotated. (1M, Serial, and A100 from `cluster-chess` cited in prose.)
  - *(b) Step breakdown (100% linear stacked vertical bars, CART):* 4 small multiples (`EPYC 48T`, `Xeon 48T`, `H100 GPU`, `H100 Hybrid`) across Chess 10M, 50M, 100M, 500M, 1B, 2B `|` HIGGS, ECBDL'14 15M (`Presort`, `Split search`, `Partition`, `Transfer`, `Other`), `▼` GPU-busy marker. C4.5 similarity + CPU Chess 500M–2B bimodal variance noted in prose/caption.
  - *Sources:* `bench/cluster-profiling/{,plain/}*/` (`profile_share_chart.py`).

### 5.5 Energy Consumption (~1.00 p: 0.68 p floats + 0.32 p prose)
- **Must show:** Whole-server energy tracks training time across 3 orders of magnitude; H100 `GPU-Based` saves 27–29% on shallow ECBDL'14 15M; `Hybrid` saves 4.0–10.5% on full-depth MNIST8M despite higher power (via shorter idle burn, though +31–41% dynamic energy).
- **Table 5 (`tab:energy`, ~0.26 p):** Existing `tab:energy` (ECBDL'14 15M d10, 5 rows) + 3-row bottom block (MNIST8M no-depth on `hopper`: `Multicore`, `GPU-Based`, `Hybrid`), columns $t$ (s), $P$ (W), $E$ (kJ), and $E_{\text{dyn}}$ ($E-\text{idle}$, kJ) for CART & C4.5. *Sources:* `bench/cluster-power/{ecbdl_15m-depth10,mnist8m-nodepth}/summary.csv`.
- **Fig. 10 (`fig:energy` / existing Fig. 6, ~0.42 p):** Keep both C4.5 stacked panels (`test-energy.pdf` + `test-energy-mnist8m.pdf`), **removing bar-end `t / P / E` numbers** (now exclusively in Table 5) so the figure focuses cleanly on visual scaling, idle vs dynamic shading, and CPU-hatched vs GPU power split.

### Page Budget Summary
| Subsection | Tables | Figures | Floats | Prose | Total |
|---|---|---|---:|---:|---:|
| 5.1 Overall Time & Hardware | T1 (0.29), T2 (0.27) | Fig. 5 bars (0.46) | 1.02 p | 0.40 p | **1.42 p** |
| 5.2 Libraries, Accuracy & Memory | T3 (0.40), T4 (0.22) | GPU timeline (0.44) | 1.06 p | 0.40 p | **1.46 p** |
| 5.3 Deep Trees (Hybrid) | — | Depth + profile (0.46) | 0.46 p | 0.35 p | **0.81 p** |
| 5.4 Scalability & Profiling | — | Scaling (0.27), Chess+prof (0.48) | 0.75 p | 0.38 p | **1.13 p** |
| 5.5 Energy Consumption | T5 (0.26) | Fig. 6 energy (0.42) | 0.68 p | 0.32 p | **1.00 p** |
| **Total** | **5 tables (1.44 p)** | **6 figures (2.53 p)** | **3.97 p** | **1.85 p** | **5.82 p** |

---

## 3. Remaining Open Questions for the Author (Max 5)

1. **A100 GPU-library runs (`final.sh -k gpulibs` on `ampere`):** Confirm we should run HIGGS d10, MNIST8M d10, *and* HIGGS d30 on `ampere` (feeding Table 1, Fig. 5 A100 ticks, and optionally `fig:depth` d30 markers).
2. **Pruned-accuracy run scope (`tab:pruned`):** Should we include the 3 pruned CPU-library rows on HIGGS (sklearn `ccp_alpha`, J48 `-C 0.25`, YaDT `-ebpg`) and compute balanced accuracy/F1 for ECBDL'14 15M (or just footnote ECBDL's 98.3% majority-class rate)?
3. **HIGGS depth-30 Hybrid threshold ($m=512$ vs $256$):** Keep each harness's recorded value (`0.847 s` in `fig:depth`, `0.821 s` in `fig:gpu-timeline`) with $m$ noted, or align `gpu-vs-others` d30 on one uniform $m$?
4. **Memory units across Sections 4–5:** Standardize on **GiB** (avoiding SOREL 4M's `75.9 GiB` becoming `81.5 GB` inside an "80 GB" GPU) and add a one-line clarification in §4.1/§4.2, or convert `tab:datasets` to GiB as well?
5. **Minor draft fixes before final prose:** Confirm correcting the abstract/intro "scales up to 40×" wording (40× is `Multicore`/`Serial Opt` on SOREL; same-code thread scaling is 34.9× on MNIST8M), §4.3's `$m$`-sweep phrasing, and `662 W` → `661 W` on MNIST8M C4.5 `GPU-Based`.

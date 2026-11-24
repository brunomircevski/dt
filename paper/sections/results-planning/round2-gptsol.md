# Round 2 — cross-critique and revised plan (`gptsol`)

## 1. Settled disagreements

1. **Time tables versus Fig. 5 — NEW-COMPROMISE.** Keep both CART/C4.5 time tables in the existing style, but remove per-bar values from Fig. 5; tables are the exact ledger, while the figure shows orders of magnitude, adjacent-step structure, and second-hardware behavior.
2. **Performance across hardware — AGREE-WITH opus.** Do not add a hardware subsection or heatmap: use dell/A100 ticks in Fig. 5, exact alternate-node entries in the time tables, and hardware curves where depth/scaling already require them.
3. **GPU-library phases — KEEP-MINE.** Retain a compact phase-timeline figure because upload/presort overlap and library binning/build costs explain the result; exact totals remain in the CART table, so the timeline has no value labels.
4. **RAM/VRAM — NEW-COMPROMISE.** Put resources beside tree fidelity in one compact, multi-block table: all-dataset memory for our backends plus HIGGS/MNIST8M library memory, rather than a separate memory chart.
5. **Chess presentation — KEEP-MINE.** Show both total ns/sample and 100%-step shares at 10M, 50M, 100M, 500M, 1B, and 2B, merged below thread scaling; totals establish crossover and shares explain it.
6. **Hybrid-wins figure — KEEP-MINE.** Use GPU-Based/Hybrid time ratio for the depth sweep, include H100 and A100, add a “who is working” panel, and add a small absolute-time HIGGS-depth-30 library panel.
7. **Energy — AGREE-WITH opus.** Keep Table energy as the exact \(t/P/E\) ledger and make Fig. 6 show idle/CPU/GPU decomposition without repeating total-value labels.
8. **Accuracy — KEEP-MINE.** Keep existing exact-versus-binned depth-10/depth-30 fidelity and request pruned full-depth results on SUSY, HIGGS, MNIST8M, ECBDL'14 15M, and SOREL 4M; timing remains unpruned.
9. **Subsection count/order — KEEP-MINE.** Use five subsections: overview, libraries/quality/resources, deep Hybrid, scalability/profile, energy; hardware is an encoding across these, not a sixth narrative.
10. **Duplicated measurements — NEW-COMPROMISE.** Give every comparison block one source: canonical `cluster-*` for the overview, co-scheduled `*-vs-others` for libraries, depth-sweep files for crossover, and power files for energy; never form a ratio across blocks.
11. **A100 libraries — AGREE-WITH opus, amended by the binding decision.** cuML/XGBoost on ampere are **NEW RUN NEEDED**, not optional or droppable; reserve their rows/panels now.
12. **Units — AGREE-WITH gemini.** Convert byte fields to GiB in Section 5 and state the unit explicitly; do not preserve the benchmark notes' ambiguous “GB” labels.

## 2. Revised complete layout

### 5.1 End-to-End Training Time and Hardware Context

- **Purpose/claims:** show the five-implementation CART/C4.5 ladder on all seven dataset entries; explain why shallow Hybrid loses; expose epyc/dell and H100/A100 behavior without a separate hardware section.
- **Table 5.x — CART training time (0.24 page):** existing `booktabs` style. Main block: seven dataset rows × Serial Simple, Serial Optimized, Multicore, GPU-Based, Hybrid from canonical `bench/cluster-{susy,higgs,mnist8m,ecbdl,sorel}` epyc/hopper summaries. Co-scheduled library block: HIGGS/MNIST8M rows for our matching baselines, scikit-learn, rpart, cuML, and XGBoost; separate H100/A100 GPU rows. Mark rpart MNIST8M failure and full-data GPU infeasibility.
- **Table 5.y — C4.5 training time (0.24 page):** same main block and style; library block contains our matching baselines, J48, YaDT-1T, and YaDT-48T. No GPU-library rows because cuML/XGBoost are CART only.
- **Figure 5 / `fig:test-bars` (0.42 page):** two panels, CART/C4.5; x = seven dataset entries, y = mean seconds on log scale; bars = five implementations on epyc/H100; thin ticks = dell for CPU bars and A100 for GPU bars; braces/arrows encode adjacent-step speed-up ranges. Remove numeric bar labels and do not add libraries: the tables own exact values and the library subsection owns external comparisons.
- **Sources:** canonical per-dataset `summary.csv` files for bars/main table; `bench/cluster-cpu-vs-others` and `bench/cluster-gpu-vs-others` only inside clearly separated comparison blocks.
- **Text allowance:** 0.35 page; subsection total about **1.25 pages**.

### 5.2 Libraries, Tree Fidelity, and Resource Cost

- **Purpose/claims:** compare CPU/GPU libraries fairly on HIGGS and MNIST8M, explain phase costs, and put speed alongside accuracy/tree shape and RAM/VRAM trade-offs.
- **Figure 5.x — GPU phase timelines (0.45 page):** 2×2 small multiples = HIGGS/MNIST8M × H100/A100 at depth 10. Rows = GPU-Based, Hybrid, cuML, XGBoost; x = elapsed seconds; segments/overlap = setup, upload, prepare/presort or binning, GPU build, and Hybrid CPU-subtree work. Use a common scale per dataset, no total-value labels, and direct phase labels only where legible. A100 panels are **NEW RUN NEEDED** through the existing `cluster-gpu-vs-others`/`final.sh -k gpulibs` harness.
- **Table 5.z — quality and resources (0.55 page, existing table style):**
  - Block A, our seven dataset entries: peak Multicore host RAM, GPU-Based host RAM/VRAM, and Hybrid host RAM/VRAM; Hybrid may share a cell with GPU-Based when rounded values coincide.
  - Block B, HIGGS/MNIST8M depth 10: each relevant external library plus our matched implementation; columns = nodes, test accuracy, host RAM, VRAM. No time column.
  - Block C, HIGGS depth 30: exact GPU-Based/Hybrid versus cuML/XGBoost; columns = nodes and test accuracy, exposing the 0.58-point binned gap and larger binned trees.
  - Block D, pruned exact trees: five datasets × CART/C4.5, with unpruned full-depth accuracy/nodes, pruned accuracy/nodes, accuracy delta, and compression ratio. This is **NEW RUN NEEDED** using one deterministic `--test-accuracy` repetition of the existing pruning protocols; ECBDL gets a majority-class footnote.
- **Sources:** `bench/cluster-{cpu,gpu}-vs-others/*/summary.csv` and JSONL phase records; canonical cluster summaries for all-dataset memory; new pruning run via `bench/tools/run.py` and `bench/tools/cluster/submit.sh`.
- **Text allowance:** 0.35 page; subsection total about **1.35 pages**.

### 5.3 Deep Trees: When Heterogeneous Execution Wins

- **Purpose/claims:** establish the crossover and then explain it through concurrent CPU/GPU work; retain deep GPU-library evidence.
- **Figure 5.y — depth, overlap, and deep libraries (0.65 page):**
  - Top row, three panels (HIGGS, MNIST8M, SOREL 4M): x = depth 5, 10, 15, 20, 25, 30, unlimited; y = GPU-Based time / Hybrid time; horizontal break-even at 1; CART/C4.5 by line style; H100/A100 by filled/open markers; direct-label unlimited ratios.
  - Bottom-left: MNIST8M H100 paired 100%-stacked bars at depth 10, 20, 30, unlimited, split into setup, GPU alone, GPU+CPU overlap, CPU alone, transfer, and host; totals and CPU thread-seconds are the only numeric annotations.
  - Bottom-right: HIGGS depth-30 absolute-time mini-panel for GPU-Based, Hybrid, cuML, XGBoost on H100 and A100. H100 uses the co-scheduled library run; A100 library bars are **NEW RUN NEEDED**. Keep this block visually separate from sweep-derived ratios because hand-off settings differ.
- **Sources:** `bench/hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-*/summary.csv` and profile JSON; depth-30 library block from `bench/cluster-gpu-vs-others`.
- **Text allowance:** 0.30 page; subsection total about **0.95 page**.

### 5.4 Thread Scaling and Cost per Sample

- **Purpose/claims:** separate thread scalability from dataset width, show the chess CPU/GPU crossover, and explain asymptotic cost through measured step shares.
- **Figure 5.z — scaling and chess profile (0.75 page):**
  - Top row: existing epyc/dell thread-scaling panels; x = 1–48 threads, y = speed-up over one thread; HIGGS/MNIST8M by color, CART/C4.5 by solid/dashed, ideal line, 24-thread socket marker, 48-thread speed-up/efficiency labels.
  - Bottom-left/middle: CART and C4.5 chess total cost; x = 10M, 50M, 100M, 500M, 1B, 2B rows (log), y = ns/sample; curves = epyc/dell Multicore and H100 GPU-Based/Hybrid; annotate crossover and 2B totals/ratios.
  - Bottom-right: CART 100%-stacked step shares at the same six sizes for epyc, dell, H100 GPU-Based, and H100 Hybrid; steps = presort, split search, partition, transfer, other; triangle = GPU-busy share. C4.5's differing total remains in its cost panel; its qualitatively similar shares stay in text.
- **Sources:** `bench/cluster-cpu-scaling/*/scaling.csv` and `bench/cluster-profiling/{plain/,}{cpu-epyc,cpu-dell,gpuonly-hopper,cuda-hopper}-chess_*`.
- **Text allowance:** 0.30 page; subsection total about **1.05 pages**.

### 5.5 Energy per Tree

- **Purpose/claims:** show that runtime dominates whole-server energy, while deep Hybrid can offset higher power through shorter execution.
- **Table 5.w — energy ledger (0.24 page):** existing ECBDL'14 15M depth-10 block plus MNIST8M unlimited block; rows = applicable implementations; CART/C4.5 columns = time, server power, whole-server energy, and energy above idle.
- **Figure 6 / `fig:energy` (0.42 page):** retain ECBDL'14 depth-10 and MNIST8M-unlimited panels, but remove total \(t/P/E\) labels. Aligned bars show idle-server, CPU-package, GPU, and residual/server components; the table owns exact totals and the figure owns decomposition.
- **Sources:** `bench/cluster-power/ecbdl_15m-depth10/summary.csv` and `bench/cluster-power/mnist8m-nodepth/summary.csv`.
- **Text allowance:** 0.25 page; subsection total about **0.90 page**.

**Estimated total:** about **5.5 LNCS pages**: 3.96 pages of floats/captions and roughly 1.5 pages of prose. Omit `test-ladder`, `test-dots`, `test-pareto`, the generic `test-profile`, and a standalone hardware heatmap; move hand-off-sensitivity detail and measurement caveats to Section 6/appendix.

## 3. Remaining questions for the author

1. Should the pruning block compare only our exact CART/C4.5 before/after pruning, or also add semantically closest pruned scikit-learn/J48 rows on HIGGS and MNIST8M?
2. For ECBDL'14, is a majority-rate warning sufficient, or should the new accuracy run also compute balanced accuracy/F1?
3. If A100 library reruns miss the paper freeze, should reserved cells remain “pending” in the draft rather than suppressing the comparison?
4. May Section 5 standardize on GiB even if the dataset table elsewhere remains in decimal GB, or should the whole paper be harmonized in the same edit?

## Five-line summary of my changed position
- I now keep both existing-style time tables and make Fig. 5 structural rather than value-labelled.
- I still retain the GPU phase timeline, but remove totals from it and add the required A100 library runs.
- I fold hardware into Fig. 5/depth/scaling encodings and combine tree fidelity with RAM/VRAM in one table.
- I adopt Opus's table/figure split for energy while retaining both absolute chess cost and phase shares.
- The revised five-subsection layout is denser, source-separated, and estimated at about 5.5 LNCS pages.

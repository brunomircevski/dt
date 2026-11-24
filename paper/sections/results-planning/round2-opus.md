# Round 2: `opus`, cross-critique and revised plan

The rule I keep from Round 1 is **each number is printed once**. "Printed" means a value label or a table cell, not the height of a bar. That rule now lets the author's tables and Fig. 5 coexist.

- The tables hold the exact depth-10 means.
- Fig. 5 drops its per-bar labels. It keeps the bar heights and adds things the tables cannot show:
  - the shape of the speed-up ladder;
  - the second node of each kind (dell, A100);
  - the libraries side by side with ours.

---

## 1. Positions on the disputed points

**1. Time tables: NEW-COMPROMISE (orchestrator's binding note)**
- Keep `tab:time-cart` and `tab:time-c45` exactly in their current style and content: 7 rows for this work (epyc / H100) and a library block for HIGGS and MNIST8M. They are the only place depth-10 means are printed.
- Fig. 5 loses its vertical value labels (`charts.py:629`). It gains:
  - hollow, hatched library bars in the HIGGS and MNIST8M groups;
  - dell and A100 ticks;
  - gptsol's step braces over adjacent implementations, carrying aggregate ranges only: "×16–35", "×14–40", "×1.1–1.7".
- I withdraw "drop the tables". gptsol and gemini also dropped one or the other, and that contradicts the request.
- gemini's "keep both with labels" fails rule 7: 70 duplicated cells.
- One fix to the current draft is needed: `tab:time-cart` is 6.1 pt overfull (gemini). Set `\tabcolsep` to 4 pt.

**2. Hardware (epyc vs dell, H100 vs A100): KEEP-MINE, extended**
- No heatmap and no hardware subsection. gemini's heatmap prints epyc and H100 values again as ratios, and it still needs a memory strip to justify its 0.37 page.
- The dell and A100 means appear in Fig. 5 as black ticks on the Serial Opt / Multicore bars and on the GPU-Based / Hybrid / cuML / XGBoost bars.
- New: a one-line **ratio row** under each dataset group (`dell ×1.4 · A100 ×1.3`, for Multicore and GPU-Based). These are numbers no table holds, so the per-dataset hardware result is printed exactly once.
- One text paragraph in 5.1 explains the pattern:
  - bandwidth: 8 vs 12 channels;
  - SUSY's A100 beats the H100, because set-up dominates;
  - ECBDL 15M on the A100 is no faster than the epyc CPUs.

**3. GPU-library phase timeline: AGREE-WITH gptsol / gemini, compacted**
- I said "text only". Both others argued that the mechanism is worth a figure:
  - the presort is hidden in the upload;
  - XGBoost bins for 2.35 s;
  - the build takes 4.5 s vs 1.43 s.
- With a 5–6 page budget, I accept.
- Changes to their version:
  - no total labels for the depth-10 lanes (the tables hold them);
  - the HIGGS depth-30 lane moves here, from my depth figure, with broken axes and totals labelled: cuML 4.18 s, XGBoost 23.8 s, GPU-Based 0.994 s, Hybrid 0.821 s. That lane is the only place those numbers appear.

**4. RAM / VRAM: KEEP-MINE (combined trees + accuracy + memory table)**
- It is close to gemini's Table 3, but it adds our 7 dataset rows. The "presorted ≈ 3× data in RAM, GPU ≈ 1× RAM + 2× VRAM" result needs all datasets, not just HIGGS and MNIST8M.
- The library block follows the author's own two-block style.
- gemini's memory bar panel would carry the same numbers a second time.

**5. Chess: NEW-COMPROMISE**
- One figure with thread scaling (all three agree on merging).
- The chess row is 4 small multiples (epyc Multicore, dell Multicore, H100 GPU-Based, H100 Hybrid; CART):
  - **absolute ns/sample, stacked by step** (mine);
  - C4.5 total as a dashed outline;
  - the epyc total dotted on the GPU panels for the crossover;
  - a ▼ marker for the GPU-busy share (gemini / gptsol);
  - one total-seconds label at 2B per panel.
- Sizes are 10M, 50M, 100M, 500M, 1B, 2B from `cluster-profiling/plain/` (totals) and the profiled runs (shares).
- 1M, epyc 1T and the A100 come from a different harness (`cluster-chess`), so they stay out (point 10). gemini's 100 % share bars for HIGGS and ECBDL stay in text: their ns/sample scale is not comparable to chess.

**6. Hybrid wins: NEW-COMPROMISE, ratio y (gptsol) + paired profile (mine)**
- 3 panels (HIGGS, MNIST8M, SOREL 4M):
  - y = GPU-Based ÷ Hybrid, with a break-even line at 1;
  - CART solid, C4.5 dashed; H100 filled, A100 hollow. That halves my 2×3 grid and makes the crossover the encoding.
- Absolute seconds go in as labels at ∞ only (printed nowhere else).
- Top axis: CART nodes per depth.
- Profile panel: MNIST8M CART on the H100, **paired** Hybrid / GPU-Based 100 % bars at depths 10, 20 and ∞ (6 segments). Pairing shows the host overhead fall from 14–18 % to 1 %, which Hybrid-only bars (gptsol) cannot.
- C4.5 and A100 profiles are text only.
- The libraries at depth 30 move to the timeline figure (point 3).

**7. Energy: KEEP-MINE**
- `tab:energy` stays, because the author named it as a liked style. It gains:
  - a block "MNIST8M, no depth limit, hopper" (3 rows);
  - an *E*−idle column (the text quotes 0.89 vs 2.27 kJ).
- Fig. 6 keeps both panels but drops its *t / P / E* value labels. It labels only what the table lacks: the idle share and the active CPU / GPU watts.
- gptsol's "drop the table" contradicts the request. gemini's "both, fully labelled" repeats 30 numbers.

**8. Accuracy: AGREE-WITH all (NEW RUN), scope from gemini + gptsol**
- Run pruned and unpruned no-limit trees, Multicore on epyc, untimed, 1 rep. `run.py --protocols cart_full_nodepth,cart_alpha_nodepth,c45_unpruned,c45 --test-accuracy`.
- Datasets: all 7 rows. Use full ECBDL and SOREL, not only the subsets: the run is untimed and the CPU fits them.
- The scikit-learn `ccp_alpha` and J48 `-C 0.25` rows run on HIGGS only (J48 on MNIST8M is 4.7 h already at depth 10).
- The exact-vs-binned evidence stays the depth-30 HIGGS rows.
- ECBDL accuracy carries a "majority class 98.32 %" dagger.

**9. Subsections: NEW-COMPROMISE, 5 subsections**
- Libraries get their own subsection (gemini, gptsol), but their times stay in 5.1's tables.
- No hardware or profiling subsection.

**10. Data sources: AGREE-WITH gptsol / gemini, plus a caption rule**
- Our cells in the tables and Fig. 5 come from `cluster-<ds>`. Library cells come from their co-scheduled jobs.
- Ours ÷ library ratios in the text use the co-scheduled ./tree run (e.g. 0.213 s, not 0.221 s). The time tables' caption says so: "ratios in the text use the ./tree run of the same job; it differs by ≤6 %".
- Depth-30 Hybrid: the timeline uses 0.821 s (default *m* = 512); the depth figure uses 0.847 s (swept *m* = 256). The swept value is slower, which breaks the "tuned = best case" claim (open question 3).

**Other conflicts**
- `test-ladder`, `test-dots`, `test-pareto`, `test-heatmap` and `test-profile` are dropped (all agree).
- `diagram-small-nodes` goes to §3.5 (all agree).
- The hand-off sweep gets two sentences in 5.3. gemini notes that it ran on the older commit `f53347a9` with set-up untimed, and that needs one caveat.
- The C4.5 chess 500M–2B bimodality (15–39 %) gets one caption clause (gptsol).
- The intro's "40× across 48 cores" is Multicore ÷ Serial Opt, not thread scaling (35×); reword it.
- The intro's "104× vs J48" mixes two jobs: the same-job ratio is 103× (gemini).

---

## 2. Revised layout

### 5.1 Training Time (≈ 1.55 p, text included)
- **`tab:time-cart` + `tab:time-c45`** (current, ≈ 0.6 p): unchanged rows. The caption adds the ratio-source sentence.
- **Fig. 5 `fig:test-bars`** (full width, 2 rows CART / C4.5, ≈ 0.55 p):
  - x: 7 dataset groups; HIGGS and MNIST8M are wider and hold the hollow library bars (CART: scikit-learn, rpart, cuML, XGBoost; C4.5: J48, YaDT 1t, YaDT 48t). rpart on MNIST8M shows "failed".
  - y: log seconds, **no value labels**.
  - **Ticks:**
    - dell on Serial Opt and Multicore;
    - A100 on GPU-Based and Hybrid;
    - A100 on cuML and XGBoost too (**NEW RUN**: `final.sh -k gpulibs` on ampere, per the author's decision).
  - **Step braces** with aggregate ranges.
  - **Ratio row** under each group: dell ÷ epyc (Multicore), A100 ÷ H100 (GPU-Based).
  - Sources: `cluster-*/{serial,cpu}-{epyc,dell}-*`, `{gpuonly,cuda}-{hopper,ampere}-*`, `cpu_other()` and `gpu_other()` in `charts.py`; YaDT needs a `STYLE` key.
- **Text:**
  - the step ranges, with the reason for each;
  - Hybrid is 3–70 % slower at depth 10 (hand-off buffer set-up ≈ 0.1 s), which leads into 5.3;
  - the hardware paragraph.

### 5.2 Existing Libraries, Tree Quality and Memory (≈ 1.0 p)
- **`tab:trees` "Trees, accuracy and memory"** (≈ 0.45 p, booktabs, two blocks like the time tables):
  - **This work** (7 rows; identical trees in all five implementations):
    - CART: nodes (d ≤ 10), acc. d ≤ 10, acc. pruned (nodes);
    - C4.5: the same three;
    - RAM (GB): Multicore, GPU-Based;
    - VRAM (GB): GPU-Based.
    - Serial Simple's RAM (≈ 1× data), Serial Opt's RAM (= Multicore ± small) and Hybrid's (±0.2 GB RAM, ±1 GB VRAM) go in one caption sentence.
  - **Existing implementations** (HIGGS | MNIST8M): scikit-learn, rpart, J48, YaDT (1t / 48t), cuML, XGBoost. Columns: nodes, acc., RAM, VRAM.
    - plus pruned rows for scikit-learn and J48 (HIGGS, NEW RUN);
    - plus "HIGGS, depth ≤ 30": exact vs cuML vs XGBoost nodes and accuracy (68.77 vs 68.19 %; 2.16 M vs 2.61 / 2.34 M).
  - Sources: `cluster-<ds>/*/summary.csv` (`nodes`, `test_accuracy`, `peak_rss`, `gpu_peak`), `cluster-cpu-vs-others`, `cluster-gpu-vs-others/summary.csv`, NEW RUN.
- **`fig:gpu-libs` timeline** (full width, 3 lanes-groups side by side or stacked, ≈ 0.4 p):
  - groups: HIGGS d10, MNIST8M d10, HIGGS d30 (broken x);
  - lanes: GPU-Based, Hybrid, cuML, XGBoost;
  - segments: set-up, upload, presort or binning, build, Hybrid CPU;
  - totals labelled **only** in the d30 group;
  - source: `cluster-gpu-vs-others/one-tree/*/results.jsonl` (`gpu_steps_chart.py` logic).
- **Text:**
  - the one-thread and 48-thread ratios against the CPU libraries (from co-scheduled runs);
  - the GPU ratios: 2.4–2.7× cuML, 2.4–3.3× XGBoost;
  - the memory trade-off (presorting costs 3× RAM; the GPU's VRAM cap is why full ECBDL and SOREL are CPU-only);
  - pruning's effect on accuracy and nodes.

### 5.3 Deep Trees: Where Hybrid Beats GPU-Based (≈ 0.9 p)
- **`fig:depth`** (full width, ≈ 0.5 p). Left, 3 panels as in point 6:
  - x = depth limit 5 … 30, ∞ (categorical);
  - y = GPU-Based ÷ Hybrid, log, with a line at 1;
  - 4 lines (CART / C4.5 × H100 / A100);
  - labels at ∞;
  - top axis: CART nodes.

  Right, spanning the full height: the paired MNIST8M H100 profile at depths 10, 20 and ∞, with total seconds on top.

  Sources: `hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-*/summary.csv` and `profile/{cuda,gpuonly}-hopper-mnist8m`.
- **Text:**
  - the crossover depths (15 / 15 / 25);
  - the no-limit speed-ups on the H100 and the A100;
  - overlap of 61 % / 74 %;
  - SOREL's thin trees as the limit;
  - the *m*-sweep in two sentences (default rule within ≈ 2 % of the tuned value).

### 5.4 Scalability in Threads and Samples (≈ 1.0 p)
- **`fig:scaling`** (full width, ≈ 0.6 p):
  - **Row (a):** the current `scaling()` on epyc and dell (height 1.25 in), with gemini's sub-NUMA tick at 12 threads on dell.
  - **Row (b):** the 4 chess small multiples from point 5.
  - Sources: `cluster-cpu-scaling/*/scaling.csv`, `cluster-profiling/plain/*` and `cluster-profiling/<mode>-<node>-chess_*`.
- **Text:**
  - thread speed-ups and efficiencies;
  - GPU per-sample cost 13 → 0.76 ns vs epyc's flat ≈ 1.7 ns, with the crossover at ≈ 10⁸ rows;
  - the 2B ratios: 2.2–2.5× vs epyc, 5.8–6.4× vs dell;
  - the step shares on HIGGS and ECBDL 15M, in one sentence each;
  - host RAM and VRAM at 2B (87 GB vs 21 + 53 GB).

### 5.5 Energy (≈ 0.9 p)
- **`tab:energy`**, extended as in point 7 (≈ 0.3 p).
- **Fig. 6 `fig:energy`**, both panels, decomposition labels only (≈ 0.45 p).
- **Text:** the current prose, trimmed of the numbers the table now carries.

**Total:** floats ≈ 3.85 p (5 tables-worth: 2 time tables, the trees table, the energy table; 5 figures: Fig. 5, the timeline, depth, scaling, Fig. 6) + text ≈ 1.8 p, so **≈ 5.6 LNCS pages**.

**NEW RUNs:**
1. pruned / unpruned accuracy (5.2);
2. cuML and XGBoost on ampere (Fig. 5 ticks);
3. optional: test accuracy along the depth sweep, for a second top axis in `fig:depth`.

---

## 3. Open questions for the author
1. **Units:** GB (consistent with Table 1 and "80 GB GPUs") or GiB (what `bench` computes)? I lean GB, converted, so Sections 1–4 need no edit.
2. **dell and A100 values:** as ticks plus a ratio row in Fig. 5 only, or should exact dell / A100 means also appear, as an appendix table?
3. **HIGGS depth 30:** the default *m* = 512 (0.821 s) beats the "tuned" *m* = 256 (0.847 s). Should we re-sweep *m* at depth 30 on the final build (cheap, `hybrid-vs-gpu-only` harness), or state that Hybrid there uses the default?
4. **Pruned library rows:** scikit-learn and J48 pruned on HIGGS only (MNIST8M J48 takes hours), or skip the library pruning and compare pruned trees for ./tree alone?
5. **ECBDL'14 accuracy:** a majority-class footnote, or a rescore for balanced accuracy (needs saved trees, or a rerun with that metric)?

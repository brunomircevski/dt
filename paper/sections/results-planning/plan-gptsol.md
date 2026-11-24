# Round 1 plan — Section 5 Results (`gptsol`)

## Editorial shape and budget

- Use five result subsections, ordered from the broad result to explanations and consequences. The narrative is: **the optimization ladder works; it beats established software; heterogeneous execution matters for deep trees; scaling/profile data explain when; speed mostly determines energy**.
- Keep the mandatory Fig. 5 and Fig. 6, add only three other full-width figures and two compact tables. Estimated visual cost is 3.4 pages; with about 1.8 pages of result text, Section 5 should cost roughly 5.2 LNCS pages.
- Report means only. Fig. 5 uses the canonical test-set runs; external-library claims use the co-scheduled comparison runs. Do not copy all Fig. 5 values into a timing table.

## 5.1 End-to-End Training Time

- **Must show:** the complete five-step implementation ladder for CART and C4.5 on all five real datasets, including the GPU-sized ECBDL'14 and SOREL subsets. This is the overview and the only place that enumerates every canonical mean time.
- **Claims:** Serial Optimized is 16–35x faster than Serial Simple; Multicore is 14–40x faster than Serial Optimized (the maximum is SOREL-20M CART); GPU-Based on H100 is 1.06–1.71x faster than Multicore on the GPU-fitting data. At depth 10, Hybrid is 3–70% slower than GPU-Based because its extra hand-off-buffer setup is not repaid.
- **Evidence:** `bench/cluster-{susy,higgs,mnist8m,ecbdl,sorel}/*-{epyc,hopper}-*/summary.csv` and the checked values in each corresponding `BENCHMARK.md`. Examples: HIGGS CART 237/7.89/0.348/0.221/0.329 s; MNIST8M 1,896/115/3.26/2.25/2.38 s; full SOREL CART 12,538/608/15.1 s.

### Figure 5 — implementation ladder (mandatory, retain `test-bars.pdf`)

- **Layout:** retain the existing two vertically aligned panels (CART, C4.5), grouped by dataset, with five colored bars and a log-time axis. Keep numeric means above bars and blanks for GPU-infeasible full ECBDL'14/SOREL. Add short braces over adjacent implementation pairs with the across-dataset speed-up range; this makes the optimization story readable without the redundant `test-ladder.pdf`.
- **Axes/encoding:** x = dataset (full and GPU subset clearly nested); y = mean training seconds, log scale; color/marker texture = implementation. Visually separate CPU bars (epyc) from GPU bars (H100) with a thin divider in each group.
- **Sources:** `bench/cluster-susy`, `cluster-higgs`, `cluster-mnist8m`, `cluster-ecbdl`, and `cluster-sorel`, specifically each run's `summary.csv`; generation starts from `paper/tools/charts.py::bars`.
- **Replaces/merges:** existing `test-bars.pdf` plus the useful speed-up annotations from `test-ladder.pdf`. It replaces both current full timing tables (`tab:time-cart`, `tab:time-c45`), whose values it already labels.
- **Cost:** 0.58 page including caption.

## 5.2 Comparison with Established CPU and GPU Libraries

- **Must show:** comparisons on HIGGS and MNIST8M only. Keep CART and C4.5 competitors distinct: scikit-learn/rpart for CART, J48/YaDT for C4.5, and cuML/XGBoost for CART only.
- **CPU claims:** on one thread Serial Optimized is 27.3x/8.8x faster than scikit-learn and 103x/147x faster than J48 on HIGGS/MNIST8M; Multicore is 620x/310x faster than scikit-learn and about 2,000x/4,900x faster than J48. Qualify YaDT on HIGGS because its 1,045-node tree is not the same C4.5 tree.
- **GPU claims:** exact GPU-Based is 2.4–2.7x faster than cuML and 2.4–3.3x faster than XGBoost at depth 10. At HIGGS depth 30, Hybrid is fastest (0.821 s), versus GPU-Based 0.994 s, cuML 4.18 s, and XGBoost 23.84 s.
- **Quality/resource claims:** exact and binned depth-10 trees have test accuracy within 0.05 percentage points, but deep HIGGS exposes a 0.58-point gap (68.77% exact versus 68.19% binned) and 8–21% more binned-tree nodes. Peak memory is a real trade-off: on MNIST8M GPU-Based uses 49 GiB VRAM, cuML 25 GiB, XGBoost 67 GiB; on CPU, our presorted Multicore uses 74 GiB RAM versus scikit-learn 24 GiB, while J48 uses 82 GiB and YaDT-48T about 20 GiB.

### Table 5.x — CPU-library comparison and memory

- **Rows:** two algorithm blocks. CART rows: Serial Optimized, Multicore, scikit-learn, rpart. C4.5 rows: Serial Optimized, Multicore, J48, YaDT-1T, YaDT-48T.
- **Columns:** implementation/threads; for each of HIGGS and MNIST8M: mean time, speed-up versus the matching one-thread library (or versus YaDT at the same thread count), nodes/test accuracy, peak host RSS. Use `--` plus a footnote for rpart's 2^31-value failure on MNIST8M.
- **Sources:** `bench/cluster-cpu-vs-others/vs-epyc-{higgs,mnist8m}/summary.csv`; Serial Simple is omitted because it answers the optimization question in Fig. 5, not the library question.
- **Style:** same grouped headers, `booktabs`, units in headers, and concise footnotes as the current CART/C4.5 tables.
- **Cost:** 0.45 page.

### Figure 5.x — cold one-tree GPU comparison

- **Layout:** three horizontal timeline panels: HIGGS depth 10, MNIST8M depth 10, HIGGS depth 30. Four rows per panel (GPU-Based, Hybrid, cuML, XGBoost). Preserve the useful broken x-axis only in the depth-30 panel.
- **Axes/encoding:** x = elapsed seconds from host-resident input to finished tree; stacked/overlapping intervals = setup, upload, prepare/presort-or-binning, GPU build, and Hybrid CPU-subtree interval. Label each row at the right with total, speed relative to GPU-Based, accuracy, nodes, and peak VRAM.
- **Sources:** `bench/cluster-gpu-vs-others/summary.csv` and `one-tree/vs-hopper-{higgs,mnist8m,higgs-depth30}/results.jsonl`; redesign `bench/cluster-gpu-vs-others/chart.png`, whose phase boundaries already carry these data.
- **Avoided duplication:** totals are not repeated in Table 5.x; Fig. 5 contains only our canonical depth-10 totals, while this figure uses the co-scheduled run (e.g. HIGGS GPU-Based 0.213 s versus canonical 0.221 s) for a fair library comparison.
- **Cost:** 0.60 page.

### Table 5.y — predictive quality with practical pruning

- **Rows:** SUSY, HIGGS, MNIST8M, ECBDL'14 15M, SOREL 4M; two compact algorithm blocks.
- **Columns per block:** unpruned full-depth test accuracy/nodes; pruned full-depth test accuracy/nodes; delta accuracy and compression ratio. No timing column: all headline timing remains the unpruned depth-10 protocol.
- **Sources:** **NEW RUN NEEDED** (details under Data gaps). State that ECBDL accuracy is dominated by its 98.32% majority class and is not evidence of useful discrimination.
- **Cost:** 0.32 page.

## 5.3 When Heterogeneous Execution Wins

- **Must show:** GPU-Based is appropriate for shallow trees, but Hybrid overtakes it once many small nodes appear and CPU subtree work overlaps GPU levels. This is the central heterogeneous-architecture result, not a footnote to profiling.
- **Claims:** on H100, no-limit Hybrid is 1.28/1.76x faster for HIGGS CART/C4.5, 1.52/1.61x for MNIST8M, and 1.18/1.08x for SOREL 4M. The crossover is around depth 15 for MNIST8M, depth 20–25 for HIGGS, and weak for SOREL's long thin trees. HIGGS depth-30 also beats both GPU libraries.
- **Evidence:** `bench/hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-{higgs,mnist8m,sorel_4m}/summary.csv`; overlap evidence from `profile/{cuda,gpuonly}-{hopper,ampere}-mnist8m/profiles/*.json` and profile `summary.csv`.

### Figure 5.y — depth crossover and CPU–GPU overlap

- **Top row (three small multiples):** HIGGS, MNIST8M, SOREL 4M. x = depth limit 5, 10, 15, 20, 25, 30, unlimited; y = GPU-Based time / Hybrid time with a horizontal break-even line at 1. Curves encode CART/C4.5 by line style and H100/A100 by filled/open markers. Direct-label the unlimited ratios.
- **Bottom row (MNIST8M on H100):** four 100%-stacked bars at depths 10, 20, 30, unlimited for Hybrid, split into setup, GPU alone, GPU+CPU overlap, CPU alone, transfer, and host. Put total seconds beside each bar and CPU subtree thread-seconds below. This explains, rather than merely restates, the crossover.
- **Sources:** the summary/profile files above; redraw rather than keep the six-panel grouped bars in `test-depth.pdf`. It also subsumes the most useful content of `chart-profile-hopper-mnist8m.png`.
- **Cost:** 0.68 page.

## 5.4 CPU Scaling and Cost per Sample

- **Must show:** CPU scaling depends on available attribute parallelism and memory bandwidth; the synthetic chess tree isolates cost per sample and extends the profile to 2x10^9 rows as required.
- **Claims:** at 48 threads, epyc reaches 22.8/18.6x on HIGGS and 34.9/32.8x on MNIST8M; dell reaches 19.1/15.9x and 25.9/23.5x. For chess, GPU fixed setup loses at 1–10M rows, H100 wins from 100M, and at 2B rows GPU-Based is 2.2x (CART) and 2.4x (C4.5) faster than epyc. CPU partition is 34–51% on large chess; GPU-busy share rises from 2–3% at 10M to about 60–64% at 2B.
- **Evidence:** scaling CSVs in `bench/cluster-cpu-scaling/cpu-{epyc,dell}-{higgs,mnist8m}/scaling.csv`; chess profile summaries and traces in `bench/cluster-profiling/{cpu-epyc,cpu-dell,gpuonly-hopper,cuda-hopper}-chess_{10m,50m,100m,500m,1b,2b}/`.

### Figure 5.z — scaling and chess profile (dense two-row figure)

- **Top row:** retain the two node panels of `test-scaling.pdf`: x = threads; y = speed-up over one thread; dataset by color/marker, CART/C4.5 by solid/dashed line, ideal line, second-socket marker, and 48-thread speed-up/efficiency labels.
- **Bottom-left and bottom-middle:** CART and C4.5 chess cost per sample. x = rows (10M–2B, log scale); y = mean training ns/sample (not total seconds), with lines for epyc Multicore, dell Multicore, H100 GPU-Based, H100 Hybrid. Annotate the CPU/GPU crossover and the 2B GPU/epyc ratio.
- **Bottom-right:** phase-share slopegraph or narrow stacked bars at 10M, 100M, 500M, 1B, 2B for epyc and H100 GPU-Based: presort, split search, partition, transfer, other; overlay a triangle for GPU-busy share. Use CART only here because C4.5 gives the same qualitative profile; its cost curve remains in the adjacent panel.
- **Replaces/merges:** `test-scaling.pdf`, the chess portion of `test-profile.pdf`, and `bench/cluster-profiling/chart-chess.png`. HIGGS/ECBDL phase bars are not plotted; their two key percentages can be stated once in text.
- **Cost:** 0.82 page.

## 5.5 Energy per Tree

- **Must show:** whole-server power is higher for faster parallel devices, but runtime dominates total energy; Hybrid's deep-tree speed-up can offset its higher power.
- **Claims:** ECBDL'14 CART/C4.5 GPU-Based uses 1.98/2.03 kJ, 27/29% less than Multicore on epyc; shallow Hybrid costs 3–5% more than GPU-Based. On no-limit MNIST8M, Hybrid is 1.52–1.58x faster while drawing 1.42–1.46x the power, yielding 4.0%/10.5% less energy than GPU-Based; Multicore uses 3.1–3.5x more energy.
- **Evidence:** `bench/cluster-power/ecbdl_15m-depth10/summary.csv` and `bench/cluster-power/mnist8m-nodepth/summary.csv`.

### Figure 6 — time, power, and energy (mandatory `fig:energy`)

- **Layout:** retain the existing paired figure: (a) ECBDL'14 15M depth 10, all five backends on their designated servers; (b) MNIST8M no limit, Multicore/GPU-Based/Hybrid all on hopper. Four aligned columns: time, whole-server power, active CPU+GPU power, whole-server energy.
- **Axes/encoding:** implementation-colored horizontal bars; lighter segment = idle-server contribution; hatch = CPU packages in GPU runs. Keep log axes for ECBDL time/energy and linear axes for MNIST8M. Put values at bar ends.
- **Sources:** the two power summaries above; existing `paper/figs/test-energy.pdf` and `test-energy-mnist8m.pdf`, generated by `paper/tools/charts.py::energy` and `energy_deep`.
- **Avoided duplication:** delete the current `tab:energy`; Fig. 6 labels the same time/power/energy values. Text reports only the comparisons and the counterintuitive above-idle qualification (Hybrid uses 31–41% more above-idle energy on MNIST8M).
- **Cost:** 0.70 page.

## Deliberately omitted or moved

- **Omit `test-ladder.pdf`:** Fig. 5 already contains every underlying time; only its adjacent-step speed-up ranges are merged into Fig. 5.
- **Omit separate hardware heatmap/performance-across-hardware subsection:** A100/H100 appears in the depth crossover, epyc/dell in scaling, and H100/epyc in energy. A four-node matrix would mostly repeat these comparisons.
- **Omit the current generic `test-profile.pdf`:** replace it with the required 10M–2B chess cost/profile and the targeted Hybrid overlap panel. Keep HIGGS/ECBDL phase findings in two sentences.
- **Move hand-off-threshold sweeps and default-rule sensitivity to Section 6 or appendix:** useful robustness evidence, but not another headline result; cite `bench/hybrid-vs-gpu-only/sweep/`.
- **Move NUMA placement, dell clock/noise investigation, the 1B partition bug history, and power-model validation to appendix/reproducibility material:** they support validity rather than answer a result question.
- **Leave `diagram-small-nodes` in Section 3:** it explains the mechanism. Section 5's depth/overlap figure supplies measured confirmation without repeating the schematic.
- **Do not report achieved bandwidth:** no final benchmark file provides a defensible achieved-versus-peak bandwidth metric.

## Data gaps and inconsistencies

- **NEW RUN NEEDED — pruning/accuracy:** use the existing cluster `bench/tools/run.py`/`bench/tools/cluster/submit.sh` harness, one deterministic accuracy run per dataset on Hybrid/H100, with protocols `cart_full_nodepth,cart_alpha_nodepth,c45_unpruned,c45`, `--test-accuracy`, and one repetition. Datasets: SUSY, HIGGS, MNIST8M, ECBDL'14 15M, SOREL 4M. Report only test accuracy, nodes, and tree hash; alpha is the existing `1e-5`, C4.5 CF the existing `0.25`. This is a cheap rerun of existing protocols, not a new benchmark design.
- Canonical HIGGS GPU-Based is 0.221 s in `cluster-higgs`, while the co-scheduled GPU-library harness gives 0.213 s. Use the former only in Fig. 5 and the latter only for library ratios; do not splice harnesses.
- HIGGS depth-30 Hybrid is 0.847 s in the swept depth benchmark but 0.821 s in the GPU-library run with a different hand-off choice. Use each only inside its own comparison and state the threshold convention in the caption.
- C4.5 on chess 500M–2B has two unexplained CPU speeds (15–39% min-to-max) although trees/work match. Means are still the standing reporting rule, but disclose this in the chess caption or validity paragraph.
- ECBDL's 98.3% accuracy is approximately the majority baseline (98.32%); never present it as evidence of predictive quality.
- Normalize memory units from byte fields to **GiB** in all new items. Several `BENCHMARK.md` summaries colloquially say GB while the comparison charts/tables use GiB.
- Full ECBDL'14 and SOREL-20M do not fit in 80 GiB VRAM; GPU and pruning-quality rows must remain explicitly labeled 15M and 4M subsets.

## Likely panel disagreements

1. **Drop the two full timing tables.** Their 70 numeric cells duplicate a mandatory, value-labeled Fig. 5; spend that space on external validity, memory, and pruning quality.
2. **Use one integrated scaling/profile figure.** A 2B chess cost-per-sample panel is more diagnostic than a standalone 1B percentage bar and directly satisfies the author's chess requirement.
3. **Give deep Hybrid its own subsection.** It is the architectural contribution and should not be buried under generic profiling, even though depth-10 Hybrid loses.
4. **Keep the GPU-library timeline.** Totals alone establish speed, but the timeline shows that exact presort overlaps upload and that binned libraries lose mostly in build/binning; that mechanism is worth 0.60 page.
5. **Request a small pruning-quality rerun.** Current `bench/` has only final unpruned test-set accuracy; implying that these depth-10 numbers represent practical pruned model quality would be misleading.

## Five-line summary
- Fig. 5 is the sole complete timing ledger and absorbs the redundant ladder/table material.
- One CPU table and one GPU timeline compare all required libraries while carrying accuracy, tree size, RAM, and VRAM.
- A redesigned depth figure makes the Hybrid crossover and concurrent CPU–GPU work the central architecture result.
- A dense scaling/profile figure extends chess cost per sample to 2B rows; Fig. 6 closes with whole-server energy.
- One cheap existing-harness rerun is needed for a defensible full-depth pruning/accuracy table.

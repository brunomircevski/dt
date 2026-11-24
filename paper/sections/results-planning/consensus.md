# Section 5 (Results) — panel consensus after Round 2

Panel: GPT-5.6 Sol (`round2-gptsol.md`), Gemini 4 Neon (`round2-gemini.md`), Claude Opus 5.5 (`round2-opus.md`). Round 1 plans: `plan-*.md`.
Status: **awaiting author decisions D1–D10 below**, then approval; no prose written.

Page estimates: gptsol 5.5, opus 5.6, gemini 5.8 LNCS pages (≈ 3.9–4.0 pages of floats).

## Agreed by all three

**Principle.** Every exact number is printed once: tables carry exact means; figures show structure (shape, ratios, decomposition) and only label values that no table holds. One data source per table/figure block, never a ratio across harnesses (canonical `cluster-<ds>` for the overview; co-scheduled `cluster-{cpu,gpu}-vs-others` for all library ratios, e.g. HIGGS GPU-Based 0.213 s there vs 0.221 s canonical; `hybrid-vs-gpu-only` for the depth sweep; `cluster-profiling` for chess; `cluster-power` for energy).

**Five subsections**, no separate hardware or profiling subsection (hardware is an encoding inside other items):

### 5.1 Training Time (≈1.3–1.5 p)
- `tab:time-cart`, `tab:time-c45` kept in current style and rows (7 dataset rows × 5 implementations, epyc/H100; library block HIGGS/MNIST8M). Fix the 6.1 pt overfull box (`\tabcolsep`). Caption: library ratios in the text use the co-scheduled ./tree run.
- **Fig. 5 (`fig:test-bars`)**, CART and C4.5 rows, log seconds, **no per-bar value labels**; adds dell ticks (Serial Opt/Multicore) and A100 ticks (GPU-Based/Hybrid), and step braces with the speed-up range of each step (×16–35, ×14–40, ×1.1–1.7). Retires `test-ladder`.
- Text: step ranges and reasons; Hybrid 3–70 % slower at depth 10 (hand-off buffer set-up) as a lead into 5.3; one hardware paragraph (dell vs epyc, A100 vs H100).

### 5.2 Existing Libraries, Tree Quality and Memory (≈1.0–1.45 p)
- **GPU-library phase timeline** figure (lanes GPU-Based, Hybrid, cuML, XGBoost; segments set-up, upload, presort/binning, build, Hybrid CPU work); no depth-10 totals (tables hold them). Shows presort hidden in upload, XGBoost binning 2.35 s, build 4.5 vs 1.43 s. Source `cluster-gpu-vs-others/one-tree/*/results.jsonl`.
- **One combined booktabs table "Trees, accuracy and memory"** in the two-block style: *This work* 7 rows (CART/C4.5 nodes and test accuracy at depth 10; peak RAM Multicore and GPU-Based; peak VRAM GPU-Based; Serial/Hybrid memory in one caption sentence); *Existing implementations* HIGGS/MNIST8M (scikit-learn, rpart, J48, YaDT 1t/48t, cuML, XGBoost: nodes, accuracy, RAM, VRAM) plus **HIGGS depth 30** exact vs binned (68.77 vs 68.19 %, 2.16 M vs 2.61/2.34 M nodes). ECBDL accuracy gets a "majority class 98.32 %" dagger. RAM/VRAM requirement is satisfied here.
- **Pruned accuracy** (untimed, NEW RUN — see D5 for placement/scope).
- Text: CPU-library ratios (one thread vs one thread, 48 vs 48), GPU ratios 2.4–2.7× cuML, 2.4–3.3× XGBoost; memory trade-off (presort ≈3× data in RAM; VRAM cap is why full ECBDL/SOREL are CPU-only); pruning effect.

### 5.3 Deep Trees: Where Hybrid Beats GPU-Based (≈0.8–0.95 p)
- **`fig:depth`** (replaces `test-depth`): HIGGS, MNIST8M, SOREL 4M; x = depth limit 5…30, ∞; CART/C4.5 by line style; **H100 filled, A100 hollow**; top axis CART nodes; plus a **"who is working" profile panel for MNIST8M CART on H100** (set-up, GPU alone, GPU+CPU overlap, CPU alone, transfer, host). Sources `hybrid-vs-gpu-only/{cuda,gpuonly}-{hopper,ampere}-*`, `profile/*-hopper-mnist8m`.
- Text: crossover depths (≈15 / 15 / 25), no-limit speed-ups on both GPUs, overlap 61 % (CART) / 74 % (C4.5), SOREL's thin trees as the limit, hand-off sweep in two sentences (with the caveat that the sweep ran on older commit f53347a9 with set-up untimed).
- This is the requested "Hybrid faster than GPU-only" case; the profiling chart comparing GPU implementations is the profile panel here plus the timeline in 5.2.

### 5.4 Scalability in Threads and Samples (≈1.0–1.1 p)
- Thread scaling epyc/dell (current `test-scaling`: HIGGS/MNIST8M, CART solid / C4.5 dashed, ideal line, socket line at 25 threads, 48-thread speed-up/efficiency labels; Gemini/Opus add a sub-NUMA tick at 12 threads on dell).
- **Chess profiling** 10M, 50M, 100M, 500M, 1B, 2B rows from `cluster-profiling` only (1M, Serial, A100 from `cluster-chess` stay in text): absolute **ns/sample** (CPU/GPU crossover ≈10⁸ rows; GPU 13 → 0.76 ns vs epyc ≈1.7 ns flat) and step breakdown with a ▼ GPU-busy marker. Encoding: see D7.
- Text: thread speed-ups (epyc 22.8/34.9×, dell 19.1/25.9×), 2B ratios (2.2–2.5× vs epyc), host RAM/VRAM at 2B, C4.5 chess 500M–2B bimodality (15–39 %) in one caption clause.

### 5.5 Energy (≈0.9–1.0 p)
- `tab:energy` kept and extended: + block "MNIST8M, no depth limit, hopper" (Multicore, GPU-Based, Hybrid) + an *E − idle* column. Sole home of exact t / P / E.
- **Fig. 6 (`fig:energy`)** both panels kept, **bar-end t/P/E labels removed**; shows only idle share and CPU/GPU split.
- Text: current prose trimmed of numbers now in the table.

### Dropped / moved (all agree)
- Dropped: `test-ladder`, `test-profile` (generic), `test-dots`, `test-pareto`, `test-heatmap`, a hardware heatmap, achieved-vs-peak bandwidth (no defensible data), standalone hand-off-sweep plot.
- `diagram-task-parallel` → §3.3, `diagram-small-nodes` → §3.5.
- NUMA placement, dell clock/noise, power-meter validation → Section 6 / reproducibility notes.

### New runs needed (all agree)
1. **Pruned vs unpruned full-depth test accuracy**, untimed, 1 rep, Multicore on epyc: `bench/tools/run.py --protocols cart_full_nodepth,cart_alpha_nodepth,c45_unpruned,c45 --reps 1 --test-accuracy` (α = 1e-5, CF = 0.25).
2. **cuML and XGBoost on ampere** (`final.sh -k gpulibs` — currently queued on hopper only): HIGGS d10, MNIST8M d10, HIGGS d30.

### Fixes to existing text found by the panel (all agree)
- Abstract/intro "scales up to 40× across 48 cores": 40× is Multicore ÷ Serial Optimized (SOREL); thread scaling tops out at 34.9× (MNIST8M). Reword.
- Intro "104× vs J48" mixes two jobs; the same-job ratio is 103×.
- Energy prose "662 W" → 661 W (MNIST8M C4.5 GPU-Based).
- §4.3 hand-off sweep wording vs the sweep's actual commit/protocol.

---

## Decisions for the author

| # | Question | Options (who) |
|---|---|---|
| D1 | Libraries inside **Fig. 5**? | (a) hollow/hatched library bars in wider HIGGS/MNIST8M groups, A100 ticks on cuML/XGBoost (opus, gemini); (b) no libraries in Fig. 5 — tables + 5.2 own them (gptsol) |
| D2 | Hardware ratio row under each Fig. 5 group (dell÷epyc Multicore, A100÷H100 GPU-Based), the only place per-dataset hardware ratios are printed | yes (opus); no, ticks + text paragraph only (gptsol, gemini) |
| D3 | GPU-library timeline layout | (a) 3 groups on H100: HIGGS d10, MNIST8M d10, HIGGS d30 with broken axis, totals labelled only at d30; A100 in text (opus, gemini); (b) 2×2 HIGGS/MNIST8M × H100/A100 at d10, d30 libraries go to `fig:depth` (gptsol) |
| D4 | HIGGS depth-30 library numbers live in | timeline (opus); depth figure mini-panel (gptsol); both, as timeline + hollow markers in depth figure (gemini) |
| D5 | Pruned accuracy: placement and scope | placement: block in the trees/memory table (gptsol) / extra columns there (opus) / separate small `tab:pruned` (gemini). Datasets: 5 incl. ECBDL 15M & SOREL 4M subsets (gptsol, gemini) / all 7 rows, full ECBDL & SOREL since untimed CPU fits (opus). Library rows: none (gptsol) / scikit-learn + J48 on HIGGS (opus) / + YaDT (gemini) |
| D6 | `fig:depth` y-axis | GPU-Based ÷ Hybrid ratio, break-even line at 1, absolute s labelled at ∞ (gptsol, opus); 2×3 grid of absolute log-time lines with Hybrid-win shading (gemini). Profile panel: paired Hybrid/GPU-Based at d10, 20, ∞ (opus, gemini) vs Hybrid-only at 10, 20, 30, ∞ (gptsol) |
| D7 | Chess encoding | (a) ns/sample lines + separate 100 % step-share bars (gptsol, gemini; gemini also adds HIGGS and ECBDL 15M share bars); (b) 4 small multiples of absolute ns/sample stacked by step, C4.5 total as outline, epyc total dotted on GPU panels (opus). Gemini's objection to (b): stacking on a log axis distorts shares (needs a linear y) |
| D8 | Thread scaling and chess | one merged figure, scaling on top, chess below (gptsol, opus); two separate figures (gemini) |
| D9 | Memory units | GiB everywhere in §5 + a note that the 80 GB GPUs hold 80 GiB (gptsol, gemini; avoids SOREL 4M 75.9 GiB reading as 81.5 GB on an "80 GB" GPU); GB converted, to match Table 1 and Sections 1–4 (opus) |
| D10 | HIGGS depth 30: the swept "tuned" m = 256 (0.847 s) is slower than the default m = 512 (0.821 s), contradicting "tuned = best case" | re-sweep m at depth 30 on the final build (cheap, `hybrid-vs-gpu-only` harness) (opus); keep both values with m stated per harness (gemini, gptsol) |

Smaller open questions raised by panelists:
- ECBDL'14 accuracy: majority-class footnote only, or rescore with balanced accuracy / F1 (needs a run that records it)?
- Exact dell / A100 means: Fig. 5 ticks only, or also an appendix table?
- If the A100 library runs miss the freeze, keep "pending" cells or drop A100 library rows?
- Optional NEW RUN: test accuracy along the depth sweep (second top axis in `fig:depth`) (opus).

---

## Author decisions (2026-11-24)

- Missing data: placeholders, visibly marked (`\tbd` in tables, a "PLACEHOLDER" watermark/hatch on chart elements).
- Text fixes done (uncommitted): abstract "scales up to 40× across 48 CPU cores" → Multicore up to 40× faster than Serial Optimized; intro library ratios from one job: scikit-learn 8.8–27×, J48 103–147× (abstract 147×); energy 662 → 661 W.
- Accuracy (pruned and binned-vs-exact) goes to a separate accuracy part, prepared in another session; Section 5 timing parts carry no accuracy columns.
- D1: Fig. 5 shows only our implementations (no library bars). Dell/A100 ticks + step braces stay.
- D2: no hardware ratio row.
- D3: GPU-library timeline must be flat (little vertical space).
- D4: HIGGS depth-30 library numbers in no table; at most on a chart.
- D6: orchestrator decides, compact.
- D7: chess drawn like the existing profiling chart (`test-profile.pdf`: horizontal 100 % bars, totals at right, ▼ GPU-busy).
- D8: merge scaling + chess only if it fits better.
- D9: units must match Sections 1–4 and no value may exceed the GPU's 80 → proposal pending.
- D10: re-sweep needed; prompt for the mordor agent in `MORDOR-PROMPT-HANDOFF-RESWEEP.md`.

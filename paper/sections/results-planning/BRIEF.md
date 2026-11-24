# Panel brief: plan Section 5 (Results) of the paper

You are one of three panelists (GPT-5.6 Sol, Gemini 4 Neon, Claude Opus 5.5) planning **Section 5 (Results)** of an LNCS paper:
"Accelerating Large-Scale Decision Tree Induction on Heterogeneous CPU--GPU Architectures" (repo: /home/bruno/Projects/dt, paper in `paper/`).

**Do NOT write paper prose.** The output is a *layout plan*: subsections, which results go where, which tables, which charts (what is plotted, axes, encodings, panels, which data files feed them), and what each subsection must show. The author approves the plan; the text is written later. Charts will be generated/adjusted later — you only specify them (a rough mock-up PNG in this folder is allowed but optional).

## Hard rules
- Do not edit anything in the repo except your own files in `paper/sections/results-planning/`. No commits, no git operations that change state.
- Do not build or run `./tree` or any benchmark locally (all runs happen on a cluster; you only read results already in `bench/`).
- Only propose results that exist in `bench/` or that are clearly a cheap rerun of an existing harness — mark the latter **NEW RUN NEEDED** and say which harness.

## Read first
- `paper/main.tex`, `paper/sections/*.tex` (Sections 1–4 are written; `results.tex` is the current draft of Section 5 — the subsection list there is **only a proposal**, adjust freely; the CART/C4.5 tables there and Table energy are in the style the author likes).
- `paper/sections/chart-tests.tex` and `paper/figs/test-*.pdf` / `paper/tools/charts.py`: candidate figures. In the compiled paper **Fig. 5 = `test-bars.pdf` (fig:test-bars)** and **Fig. 6 = `test-energy*.pdf` (fig:energy)**. Other candidates: test-ladder, test-scaling, test-profile, test-depth, diagram-small-nodes (Fig. 8, may belong to Section 3).
- `bench/README.md` and each `bench/*/BENCHMARK.md` (+ chart PNGs, summary/CSV/JSON files) — the available results:
  cluster-{susy,higgs,ecbdl,mnist8m,sorel} (all backends on epyc/dell/hopper/ampere), cluster-chess (1M–1B rows), cluster-cpu-scaling (1–48 threads), cluster-cpu-vs-others (scikit-learn, rpart, J48, YaDT), cluster-gpu-vs-others (cuML, XGBoost, incl. per-phase breakdown, depth 30), cluster-profiling (step shares incl. chess up to 2B), cluster-power (energy), hybrid-vs-gpu-only (depth sweep, hand-off sweep, profiling).
  Recorded per run (check the files): time, tree size, peak host RAM, peak VRAM, test accuracy where trees are compared.

## Author's requirements (must be satisfied)
1. Fig. 5 (test-bars) is included and lives in Section 5. Fig. 6 (energy) goes in the energy/power subsection.
2. A case where **Hybrid is faster than GPU-only** (deep trees; e.g. test-depth / hybrid-vs-gpu-only, profiling of GPU implementations vs others).
3. The **chess profiling** chart (cost per sample, scaling to 2·10⁹ rows).
4. **Comparison with CPU libraries** (scikit-learn, rpart, J48, YaDT) and with GPU libraries (cuML, XGBoost).
5. **Accuracy comparison** — e.g. with pruning — but most timing is unpruned.
6. **RAM / VRAM usage** shown somewhere.
7. Generally include everything valuable, **don't repeat** the same numbers in several places.
8. The author wants **dense charts** carrying a lot of information (multi-panel, small multiples, dual encodings, annotations) and **tables in the style of the existing ones**. Propose new charts or merged/redesigned ones where that beats the current candidates.

## Standing author decisions (do not contradict)
- GPU libraries are CART only (no cuML entropy); C4.5 has no GPU competitor. Library comparisons on HIGGS and MNIST8M only. Airline is out; chess is used for profiling/scaling only.
- Timing = unpruned trees, depth 10 for most benches; accuracy comparison may use pruned trees (untimed).
- Report means only (no medians / CIs). Fresh process per run, no warm-up.
- Hybrid's hand-off threshold is tuned per dataset/GPU/depth (best case) on purpose.
- Section 4 has no results; all results live in Section 5 (and discussion in Section 6).
- LNCS page budget is tight: prefer few dense figures over many sparse ones; say roughly how much space each item costs.

## Deliverable
Round 1: write `paper/sections/results-planning/plan-<you>.md` (`<you>` = `gptsol`, `gemini`, or `opus`) containing:
1. Proposed subsection list (order, title, 1–3 bullet "what it must show", key claims supported by which numbers/files).
2. Every table (columns/rows, source files, where it sits) and every figure (panels, axes, encodings, annotations, data source files, existing chart it replaces/merges), with an estimated page cost.
3. What you deliberately leave out or move to Section 6 / appendix, and why.
4. Data gaps: anything required that is not in `bench/` yet (NEW RUN NEEDED) and any inconsistency you found in the data.
5. A short list of the 3–5 decisions you expect the other panelists to disagree with, with your argument.

Be concrete (file paths, numbers you checked). Keep the plan under ~250 lines.

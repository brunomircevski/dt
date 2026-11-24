# Round 2: cross-critique and revised plan

All three Round 1 plans are in this folder: `plan-gptsol.md`, `plan-gemini.md`, `plan-opus.md`. Read all three (and `BRIEF.md` again).

## Orchestrator notes (from the author's request, binding)
- The author asked for "results in **tables similar to existing ones**" (the CART/C4.5 time tables in `paper/sections/results.tex` and Table energy). Two plans drop both time tables in favour of a value-labelled Fig. 5. That contradicts the request unless you find a resolution that keeps tables in that style. Resolve the Fig. 5 ↔ tables redundancy another way (e.g. Fig. 5 without per-bar value labels, showing speed-up structure / extra hardware / libraries, while the tables carry the exact numbers; or tables carry what Fig. 5 cannot). Agree on one answer.
- Fig. 5 (`test-bars`) must stay in Section 5; Fig. 6 (energy) in the energy subsection. Whether Table energy stays alongside Fig. 6 is open — decide.
- The author wants dense, information-rich charts and "generally everything included, without repeating too much". Page budget is real but secondary to content; aim for roughly 5–6 LNCS pages for the section.
- Author decision from earlier: cuML and XGBoost should run on both hopper and ampere (only hopper data exists now) — treat A100 library rows as NEW RUN NEEDED, not as dropped.

## Points of disagreement to settle (at least)
1. Time tables kept / merged / dropped; what Fig. 5 shows then.
2. Separate "Performance across hardware" content (epyc vs dell, H100 vs A100): own figure/heatmap, ticks in Fig. 5, or spread over other items?
3. GPU-library phase timeline: own figure vs text only vs merged into another figure.
4. Where RAM/VRAM goes (library table, own panel, combined tree/memory table).
5. Chess: absolute ns/sample vs step shares vs both; which sizes; merged with thread scaling or not.
6. Hybrid-wins figure: depth sweep ratio vs absolute times; include profiling ("who is working") panel; include A100; include GPU libraries at depth 30.
7. Energy: table + Fig. 6, or Fig. 6 only, or table holds numbers and Fig. 6 shows the split.
8. Accuracy: pruned-accuracy table (NEW RUN) vs exact-vs-binned only; which datasets.
9. Number and order of subsections.
10. Data-source rule for duplicated measurements (e.g. HIGGS GPU-Based 0.221 vs 0.213 s).

## Deliverable
Write `paper/sections/results-planning/round2-<you>.md` (`<you>` = gptsol / gemini / opus), under ~200 lines:
1. For each point above (and any other conflict you see): your position now, conceding where another plan is better, with a one-line reason. Mark each `AGREE-WITH <name>`, `KEEP-MINE`, or `NEW-COMPROMISE`.
2. Your **revised complete layout**: subsections in order; for each, the tables and figures (panels/axes/encodings/sources) and page cost; total page estimate.
3. Remaining open questions for the author (max 5).

Do not edit any other file, do not commit, do not build or run benchmarks.

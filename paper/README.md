# Paper

Working title: *Accelerating Large-Scale Decision Tree Induction on
Heterogeneous CPU-GPU Architectures*. Springer LNCS format (`llncs.cls` 2.24,
`splncs04.bst`, from Springer's LaTeX2e proceedings template).

| File | Content |
|---|---|
| `main.tex` | The paper: section headers, blue `\plan{...}` placeholders saying what goes where, the dataset table |
| `references.bib` | Bibliography (written from memory: verify every entry) |

The figures are temporary: `\graphicspath` points at `../bench/results/`, so
they are the benchmarks' PNGs and change when the benchmarks are rerun.

```bash
cd paper && latexmk -pdf main.tex
```

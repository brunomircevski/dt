# References for the paper

*Accelerating Large-Scale Decision Tree Induction on Heterogeneous CPU-GPU
Architectures* (working title, `paper/main.tex`).

Every reference given so far, the PDFs that were already in `refs/`, and the
useful entries of `paper/references.bib`, in one place. Each entry has:

* **Citation**: checked against the PDF or the publisher where possible.
* **PDF**: the file in this directory, or why there is none and where to get it.
* **Original note**: the description from the list you gave, kept as written (PL or EN).
* **Our view**: what the work does, in the context of `./tree` (exact CART and
  C4.5, presorted columns, byte-identical trees on the `--simple`, `--serial`,
  `--parallel`, `--cuda` and `--gpu-only` backends; the hybrid grows large nodes
  on the GPU level by level and hands small subtrees to the CPU thread pool)
  and of our benchmarks (SUSY, HIGGS, ECBDL'14, MNIST8M, SOREL-20M; up to 32M
  rows, 2381 features, 121 GB; against scikit-learn, rpart, Weka J48, YaDT,
  cuML and XGBoost).
* **Usefulness**: ★★★★★ must cite and read closely · ★★★★ should cite ·
  ★★★ cite in passing · ★★ optional · ★ skip.
* **Use in paper**: where it goes and what to say.

PDF names are `Short-Article-Title_FirstAuthorYear.pdf`.

## Quick index

| # | Short name | Year | Topic | ★ | PDF |
|---|---|---|---|---|---|
| 1 | Strnad & Nerat, GPU classification trees | 2016 | single tree, GPU (CART) | ★★★★★ | yes |
| 2 | Lo et al., CUDT | 2014 | single tree, GPU (SPRINT) | ★★★★★ | yes |
| 3 | Nasridinov et al., DT on GPU | 2014 | single tree, GPU (ID3), maybe CPU+GPU | ★★★★★ | **no** (paywalled) |
| 4 | Liao et al., CudaTree | 2013 | random forest, GPU, CPU+GPU forest | ★★★★ | **no** (link dead) |
| 5 | Sharp, DT and forests on GPU | 2008 | forests, GPU (vision) | ★★★ | yes |
| 6 | Aldinucci et al., FastFlow (ECML-PKDD) | 2010 | single tree, multicore (C4.5) | ★★★★ | yes |
| 7 | Aldinucci et al., FastFlow (CCPE) | 2014 | single tree, multicore (C4.5) | ★★★★★ | yes |
| 8 | Shafer et al., SPRINT | 1996 | presorted attribute lists, parallel | ★★★★★ | yes |
| 9 | Mehta et al., SLIQ | 1996 | presorting, scalable | ★★★★ | yes |
| 10 | Ruggieri, Efficient C4.5 | 2002 | fast C4.5 (EC4.5/YaDT) | ★★★★ | **no** (paywalled) |
| 11 | Abuzaid et al., Yggdrasil | 2016 | deep trees, distributed, exact | ★★★★★ | yes |
| 12 | Loreti & Visani, parallel GLEAMS | 2024 | parallel DT taxonomy, Ray | ★★★ | yes |
| 13 | Chattratichat et al., large-scale DM | 1997 | parallel DM, data/task parallelism | ★★ | yes |
| 14 | Chen & Guestrin, XGBoost | 2016 | GBDT system | ★★★★★ | yes |
| 15 | Mitchell & Frank, XGBoost on GPU | 2017 | GBDT, GPU, exact split | ★★★★★ | yes |
| 16 | Zhang, Si, Hsieh, GPU tree boosting | 2018 | GBDT, GPU histograms | ★★★★ | yes |
| 17 | Wen et al., GPU-GBDT (IPDPS) | 2018 | GBDT, GPU, exact | ★★★★ | yes |
| 18 | Wen et al., ThunderGBM (TPDS) | 2019 | GBDT, GPU, multi-GPU | ★★★★ | yes |
| 19 | Liu, Peng, Wen, multi-output GBDT | 2025 | GBDT, GPU, ICPP | ★★★ | yes |
| 20 | Ke et al., LightGBM | 2017 | GBDT, histograms | ★★★ | yes |
| 21 | Wang et al., XGT | 2025 | private DT on GPU (MPC) | ★★ | yes |
| 22 | Costa & Pedreira, DT survey | 2023 | survey | ★★★ | **no** (paywalled) |
| 23 | Mienye & Jere, DT survey | 2024 | survey | ★★★ | yes |
| 24 | Rivera-Lopez et al., metaheuristic DT survey | 2022 | survey (evolutionary DT) | ★★ | yes |
| 25 | Breiman et al., CART | 1984 | algorithm | ★★★★★ | book |
| 26 | Quinlan, C4.5 | 1993 | algorithm | ★★★★★ | book |
| 27 | Fayyad & Irani, continuous attributes | 1992 | boundary-point shortcut (used in our code) | ★★★★ | **no** (OA, bot-blocked) |
| 28 | Pedregosa et al., scikit-learn | 2011 | baseline | ★★★★ | yes |
| 29 | Hall et al., WEKA | 2009 | baseline (J48) | ★★★★ | yes |
| 30 | Therneau & Atkinson, rpart | — | baseline | ★★★ | software |
| 31 | Raschka et al., cuML | 2020 | baseline | ★★★★ | **no** (OA, not fetched) |
| 32 | Merrill & Garland, decoupled look-back scan | 2016 | GPU primitive | ★★★ | yes |
| 33 | Baldi et al., HIGGS/SUSY | 2014 | dataset | ★★★★ | yes |
| 34 | Triguero et al., ECBDL'14 | 2015 | dataset | ★★★★ | **no** (OA, bot-blocked) |
| 35 | Loosli et al., MNIST8M | 2007 | dataset | ★★★★ | book chapter |
| 36 | Harang & Rudd, SOREL-20M | 2020 | dataset | ★★★★ | yes |

---

## Before writing: the "first heterogeneous CPU-GPU" claim

The notes say *"as far as we know, it is the first such approach"*. Three
references need checking before the paper says it:

* **Nasridinov et al. 2014 (#3)**: its abstract says the method *"outperforms
  purely GPU-based implementation and CPU-based sequential implementation"*,
  which suggests the method is not purely on the GPU and may mix the two
  devices. We have no PDF. Get it (library access, DOI below) and read it
  before claiming novelty.
* **CudaTree (#4)**: ships a hybrid random forest that grows some trees with
  scikit-learn on the CPU cores and the others on the GPU at the same time.
  That is CPU+GPU at the **tree** level (whole trees per device), not at the
  **node** level as in our work. It also switches *on the GPU* from
  breadth-first (data-parallel) to depth-first (task-parallel) when nodes get
  small, which is close to our size-based split. It must be cited and the
  difference stated.
* **CUDT (#2)**: CPU only controls; GPU computes. Our notes already contrast
  with this ("unlike approaches in which the CPU primarily controls GPU
  execution").

A safe wording: *"To our knowledge, no previous work assigns the nodes of a
single CART or C4.5 tree to the CPU or the GPU by their size, with both devices
growing the tree."* Also check the co-author's own GPU and hybrid work on
evolutionary tree induction (Jurczuk, Czajkowski, Kretowski), cited in the
Rivera-Lopez survey (#24); a reviewer from that community will know it.

## Missing PDFs (get them manually)

| # | Paper | Why missing | Where |
|---|---|---|---|
| 3 | Nasridinov et al. 2014 | paywalled (Springer) | https://doi.org/10.1007/s00607-013-0343-z |
| 4 | Liao et al. 2013, CudaTree | biglearn.org file gone, Wayback blocked scripts | https://hgpu.org/?p=11070 or https://www.academia.edu/5330891 (in a browser) |
| 10 | Ruggieri 2002, EC4.5 | paywalled (IEEE) | https://doi.org/10.1109/69.991727 |
| 22 | Costa & Pedreira 2023 | paywalled (Springer) | https://doi.org/10.1007/s10462-022-10275-5 |
| 27 | Fayyad & Irani 1992 | open access, Springer blocks scripts | https://link.springer.com/content/pdf/10.1007/BF00994007.pdf (in a browser) |
| 31 | Raschka et al. 2020 (cuML) | open access (MDPI), not fetched | https://doi.org/10.3390/info11040193 |
| 34 | Triguero et al. 2015 | open access, repository blocks scripts | https://nottingham-repository.worktribe.com/preview/982001/Triguero-et-al-cleanVersion.pdf (in a browser) |

The two CUDT files were the same article (Hindawi and Wiley downloads), one is
kept.

---

## 1. Single decision trees on GPUs

### 1. Strnad & Nerat, Parallel construction of classification trees on a GPU

* **Citation**: D. Strnad, A. Nerat. Parallel construction of classification
  trees on a GPU. *Concurrency and Computation: Practice and Experience*
  28(5):1417–1436, 2016 (online 2015). DOI 10.1002/cpe.3660.
* **PDF**: `Parallel-Construction-Classification-Trees-on-GPU_Strnad2016.pdf`
* **Original note (PL)**: Jedna z najważniejszych do przeczytania: budowa
  binarnych drzew w CUDA, równoległość na poziomie węzłów, cech i podziałów;
  porównanie z sekwencyjną oraz wielowątkową wersją CPU.
* **Our view**: The closest prior work. A CART-like binary tree (numeric
  thresholds plus categorical bipartitions) built fully on the GPU (GTX 780,
  3 GB) with node-, attribute- and split-level parallelism. Nodes are batched
  from a queue so that the largest task in a batch has at most about twice the
  rows of the smallest, which is a size-aware scheduling of nodes, but all on
  the GPU. Compared against its own sequential and multi-threaded CPU versions
  (order-of-magnitude speed-up); datasets are UCI-size, far below ours. Its
  related-work section summarises earlier CPU speed-ups (3× to 7× on
  multicores) and earlier GPU trees, a good source of further citations.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (Decision trees on GPUs): the main GPU
  baseline in the literature. In the method section contrast its batching of
  nodes by size on one device with our size threshold between two devices.
  In the discussion, note that it ran on datasets of thousands to a few
  million rows and 3 GB of VRAM, while we train up to 15M x 631 on an 80 GB
  GPU and 32M x 631 on the CPU. Its observation that tree building matters
  inside iterative methods (boosting) supports our motivation.

### 2. Lo et al., CUDT: A CUDA Based Decision Tree Algorithm

* **Citation**: W.-T. Lo, Y.-S. Chang, R.-K. Sheu, C.-C. Chiu, S.-M. Yuan.
  CUDT: A CUDA Based Decision Tree Algorithm. *The Scientific World Journal*
  2014, Article 745640, 12 pages. DOI 10.1155/2014/745640.
* **PDF**: `CUDT-CUDA-Based-Decision-Tree_Lo2014.pdf`
* **Original note (PL)**: Bardzo bliska tematycznie: budowa drzewa w CUDA, CPU
  steruje wykonaniem, GPU realizuje obliczenia. Rozwiązanie wywodzi się ze
  SPRINT; porównania obejmują Weka J48 i SPRINT.
* **Our view**: SPRINT's attribute lists moved to CUDA: the CPU drives the
  tree, the GPU finds split points and splits the lists. Reports 5–55×
  over Weka J48 and up to 18× over SPRINT on a GeForce 9800GT. Same family as
  our presorted columns (sort once, keep sorted lists per node). The CPU does
  no tree work, which is exactly the contrast our notes draw. Old hardware and
  small data make the absolute numbers irrelevant today.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (GPU): the "CPU controls, GPU computes"
  design that our hybrid departs from. Also the precedent for comparing
  against Weka J48, which we do too.

### 3. Nasridinov, Lee, Park, Decision tree construction on GPU

* **Citation**: A. Nasridinov, Y. Lee, Y.-H. Park. Decision tree construction
  on GPU: ubiquitous parallel computing approach. *Computing* 96:403–413,
  2014 (online 2013). DOI 10.1007/s00607-013-0343-z.
* **PDF**: none (paywalled), see *Missing PDFs*.
* **Original note (PL)**: Równoległa budowa ID3 na GPU, z uwzględnieniem
  czasu oraz zużycia energii. Bliska problematyka, choć inny algorytm
  indukcji.
* **Our view**: ID3 parallelised at two levels: node-by-node tree building and
  sorting the records inside a node. The abstract claims it beats both a
  purely GPU implementation and a sequential CPU one, so the method may itself
  combine the CPU and the GPU (see the novelty section above). It also
  measures energy, which matches our power benchmark (`bench/power`).
* **Usefulness**: ★★★★★ (because of the novelty risk; the method itself is old
  and small-scale)
* **Use in paper**: Related work (GPU). Read before writing the "first
  heterogeneous" sentence. If we report energy, cite it as earlier work that
  measured energy of GPU tree induction.

### 4. Liao, Rubinsteyn, Power, Li, Learning Random Forests on the GPU (CudaTree)

* **Citation**: Y. Liao, A. Rubinsteyn, R. Power, J. Li. Learning Random
  Forests on the GPU. Big Learning Workshop, NIPS 2013. (`liao2013cudatree`
  in `paper/references.bib`.)
* **PDF**: none (link dead), see *Missing PDFs*.
* **Original note**: — (from `paper/references.bib`)
* **Our view**: CudaTree: random forest on the GPU that adaptively switches
  between data parallelism (breadth-first over many nodes) and task
  parallelism (depth-first per node) as nodes get small. Faster than
  scikit-learn and wiseRF on larger datasets. The package also has a hybrid
  forest that grows trees on the GPU and in scikit-learn on the CPU at the same
  time (per the CudaTree package; confirm in the paper once the PDF is found). Both ideas are close to ours; neither splits one tree's nodes between
  the devices.
* **Usefulness**: ★★★★
* **Use in paper**: Related work (GPU) and the novelty statement: the closest
  "CPU+GPU" tree work we know; say it splits whole trees of a forest between
  devices, while we split the nodes of one tree by size.

### 5. Sharp, Implementing Decision Trees and Forests on a GPU

* **Citation**: T. Sharp. Implementing Decision Trees and Forests on a GPU.
  In *Computer Vision – ECCV 2008*, LNCS 5305, pp. 595–608, Springer, 2008.
  DOI 10.1007/978-3-540-88693-8_44.
* **PDF**: `DT-and-Forests-on-GPU_Sharp2008.pdf`
* **Original note (PL)**: Wczesna praca o uczeniu i wykonywaniu drzew
  całkowicie na GPU. Kontekst rozpoznawania obiektów, więc mniej bezpośrednie
  porównanie z klasycznym CART/C4.5 na danych tabelarycznych.
* **Our view**: The first GPU decision trees: forests for object recognition
  trained and evaluated with Direct3D shaders (pre-CUDA), histograms of
  random feature responses, about 100× over the CPU. Random features and
  histogram thresholds, not exhaustive search over sorted values; images, not
  tabular data.
* **Usefulness**: ★★★
* **Use in paper**: One sentence in related work: GPU trees date back to 2008,
  first in computer vision.

## 2. Single decision trees on multicore CPUs and clusters

### 6. Aldinucci, Ruggieri, Torquati, Porting Decision Tree Algorithms to Multicore using FastFlow

* **Citation**: M. Aldinucci, S. Ruggieri, M. Torquati. Porting Decision Tree
  Algorithms to Multicore Using FastFlow. In *ECML PKDD 2010*, LNCS 6321,
  pp. 7–23, Springer, 2010. DOI 10.1007/978-3-642-15880-3_7.
  arXiv:1006.3424.
* **PDF**: `Porting-DT-Algorithms-to-Multicore-FastFlow_Aldinucci2010.pdf`
  (arXiv version)
* **Original note (PL)**: Równoległa implementacja C4.5 na wielordzeniowym
  CPU. Bardzo istotny punkt odniesienia dla Waszej wersji CPU. Autorzy
  raportują przyspieszenie do około 7× na ośmiu rdzeniach.
* **Our view**: YaDT (Ruggieri's C4.5) parallelised with FastFlow's
  farm/feedback skeletons: nodes as tasks (NP), and attributes of a node as
  tasks (NAP, Nodes & Attributes Parallelisation). This mix of node- and attribute-level parallelism
  is the same as our `--parallel` backend (features of a big node in
  parallel, subtrees as tasks). Up to 7× on 8 cores.
* **Usefulness**: ★★★★ (cite with #7; #7 is the full version)
* **Use in paper**: Related work (CPU). Cite together with #7.

### 7. Aldinucci, Ruggieri, Torquati, Decision tree building on multi-core using FastFlow

* **Citation**: M. Aldinucci, S. Ruggieri, M. Torquati. Decision tree building
  on multi-core using FastFlow. *Concurrency and Computation: Practice and
  Experience* 26(3):800–820, 2014 (online 2013). DOI 10.1002/cpe.3063.
* **PDF**: `DT-Building-on-Multicore-FastFlow_Aldinucci2014.pdf` (author's
  manuscript, AperTO Turin repository)
* **Original note (PL)**: Rozwinięte badanie równoległego C4.5, z analizą
  ograniczeń czasowych i efektywności przyjętych sposobów zrównoleglenia.
* **Our view**: The extended study: NP and NAP strategies, cost models for
  when to parallelise a node's attributes, evaluation on the largest UCI sets
  (Census-Income, Forest Cover, KDD Cup 99, p53Mutants) and a 10M-row
  synthetic set, speed-ups up to about 7×. Our multicore numbers
  (48 threads, ECBDL in 13.5 s, 37× over our serial) can be put next to
  theirs. Its analysis of what limits speed-up (sorting, small nodes, memory)
  matches our finding that the multicore code is memory-bandwidth bound.
  YaDT is also one of our CPU baselines.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (CPU): the reference parallel C4.5. Method:
  our CPU backend uses the same node/attribute decomposition; say what is
  different (presorted columns, out-of-place partition, slot memory layout).
  Results: compare scaling.

### 8. Shafer, Agrawal, Mehta, SPRINT

* **Citation**: J. Shafer, R. Agrawal, M. Mehta. SPRINT: A Scalable Parallel
  Classifier for Data Mining. In *Proc. 22nd VLDB*, pp. 544–555, 1996.
  (`shafer1996sprint`)
* **PDF**: `SPRINT-Scalable-Parallel-Classifier_Shafer1996.pdf`
* **Original note**: — (from `paper/references.bib`)
* **Our view**: Presorted attribute lists, one per attribute, split with the
  node; distributed over processors (shared-nothing). The ancestor of our
  presorted columns and of CUDT (#2). Exact splits, no binning.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (CPU) and method: origin of presorting with
  attribute lists that are partitioned, not resorted, at each node.

### 9. Mehta, Agrawal, Rissanen, SLIQ

* **Citation**: M. Mehta, R. Agrawal, J. Rissanen. SLIQ: A Fast Scalable
  Classifier for Data Mining. In *EDBT'96*, LNCS 1057, pp. 18–32, Springer,
  1996. DOI 10.1007/BFb0014141. (`mehta1996sliq`)
* **PDF**: `SLIQ-Fast-Scalable-Classifier_Mehta1996.pdf`
* **Original note**: — (from `paper/references.bib`)
* **Our view**: Presorting once plus breadth-first growth with a class list;
  the first scalable exact-split classifier. Our GPU grows large nodes
  breadth-first (level by level), as SLIQ did.
* **Usefulness**: ★★★★
* **Use in paper**: Related work, with SPRINT: presorting and breadth-first
  growth.

### 10. Ruggieri, Efficient C4.5

* **Citation**: S. Ruggieri. Efficient C4.5. *IEEE Transactions on Knowledge
  and Data Engineering* 14(2):438–444, 2002. DOI 10.1109/69.991727.
  (`ruggieri2002yadt`; the YaDT program itself is described in Ruggieri,
  "YaDT: Yet another Decision tree builder", ICTAI 2004, pp. 260–265.)
* **PDF**: none (paywalled), see *Missing PDFs*.
* **Original note**: — (from `paper/references.bib`)
* **Our view**: EC4.5: faster threshold search for C4.5 (binary search,
  counting sort, presorting) with the same tree as Quinlan's C4.5. YaDT,
  its successor, is a baseline in our CPU benchmark. Note: the bib key says
  "yadt" but the paper is EC4.5; if YaDT is the baseline, cite the ICTAI 2004
  paper too.
* **Usefulness**: ★★★★
* **Use in paper**: Compared implementations (YaDT); related work on fast
  sequential C4.5.

### 11. Abuzaid et al., Yggdrasil

* **Citation**: F. Abuzaid, J. K. Bradley, F. T. Liang, A. Feng, L. Yang,
  M. Zaharia, A. S. Talwalkar. Yggdrasil: An Optimized System for Training
  Deep Decision Trees at Scale. In *Advances in Neural Information Processing
  Systems 29 (NIPS 2016)*, 2016.
* **PDF**: `Yggdrasil-Deep-DTs-at-Scale_Abuzaid2016.pdf`
* **Original note (PL)**: Rozproszone uczenie drzew, podział danych według
  cech, kompresja i ograniczanie komunikacji. Bez binowania progów, z
  eksperymentami m.in. na MNIST8M — dlatego szczególnie przydatne dla Waszego
  „large-scale".
* **Our view**: Distributed (Spark) exact tree training with vertical
  (by-feature) partitioning, which beats the row-partitioned PLANET approach
  of MLlib and XGBoost as depth and dimensionality grow: up to 6× on MNIST8M
  and 24× on a 2M x 3500 Yahoo set for depth > 10. Same dataset (MNIST8M) and
  same exact-split philosophy as ours; our by-feature column layout is the
  shared-memory analogue of its vertical partitioning.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (CPU/distributed) and the "large-scale"
  argument: exact splits on millions of rows and thousands of features,
  MNIST8M. Possibly compare our MNIST8M single-machine times with their
  cluster times (different depth and hardware: state them).

### 12. Loreti & Visani, Parallel approaches for a decision tree-based explainability algorithm

* **Citation**: D. Loreti, G. Visani. Parallel approaches for a decision
  tree-based explainability algorithm. *Future Generation Computer Systems*
  158:308–322, 2024. DOI 10.1016/j.future.2024.04.044.
* **PDF**: `Parallel-Approaches-DT-Explainability-GLEAMS_Loreti2024.pdf`
  (was in `refs/`)
* **Original note**: — (already in `refs/`)
* **Our view**: Parallelises GLEAMS, a tree-based explainability method, with
  Ray. Its related-work section is a recent overview of parallel tree
  building (a table classifying SPRINT, ScalParC, PLANET and others by
  data/task parallelism and partitioning) and argues parallelism also cuts energy. The algorithm itself
  is not CART/C4.5.
* **Usefulness**: ★★★
* **Use in paper**: Related work: a recent paper (2024, FGCS) showing the
  topic is still active; borrow its taxonomy of parallel strategies to place
  ours (task-parallel subtrees + data-parallel nodes + device split).

### 13. Chattratichat et al., Large Scale Data Mining: Challenges and Responses

* **Citation**: J. Chattratichat, J. Darlington, M. Ghanem, Y. Guo,
  H. Hüning, M. Köhler, J. Sutiwaraphun, H. W. To, D. Yang. Large Scale Data
  Mining: Challenges and Responses. In *Proc. KDD-97*, AAAI, 1997 (4 pages).
* **PDF**: `Large-Scale-Data-Mining-Challenges-Responses_Chattratichat1997.pdf`
  (was in `refs/`)
* **Original note**: — (already in `refs/`)
* **Our view**: Early parallel data mining (Imperial College): classification
  trees parallelised by data parallelism (training set split across
  processors), task parallelism (subtrees as tasks) and hybrids, chosen with
  algorithmic skeletons and performance models. Historical; the task/data
  terminology is still the one we use.
* **Usefulness**: ★★
* **Use in paper**: Optional, one citation for "data- and task-parallel tree
  induction has been studied since the 1990s".

## 3. Tree boosting systems and GPU GBDT

### 14. Chen & Guestrin, XGBoost

* **Citation**: T. Chen, C. Guestrin. XGBoost: A Scalable Tree Boosting
  System. In *Proc. 22nd ACM SIGKDD*, pp. 785–794, ACM, 2016.
  DOI 10.1145/2939672.2939785. (`chen2016xgboost`)
* **PDF**: `XGBoost-Scalable-Tree-Boosting_Chen2016.pdf` (arXiv version)
* **Original note (EN)**: Chen, T., & Guestrin, C. (2016, August). XGBoost: A
  scalable tree boosting system. In Proceedings of the 22nd acm sigkdd
  international conference on knowledge discovery and data mining
  (pp. 785-794). ACM.
* **Our view**: The GBDT system: exact greedy and approximate (quantile
  sketch) split finding, column blocks (presorted, compressed columns: the
  same idea as our presorted columns), cache-aware access, sparsity. One of
  our GPU baselines (`gpu_hist`, one tree, depth-limited).
* **Usefulness**: ★★★★★
* **Use in paper**: Introduction (ensembles motivate fast tree building),
  related work, compared implementations (why XGBoost with squared error on
  one tree is the closest equivalent to a Gini tree on bins).

### 15. Mitchell & Frank, Accelerating the XGBoost algorithm using GPU computing

* **Citation**: R. Mitchell, E. Frank. Accelerating the XGBoost algorithm
  using GPU computing. *PeerJ Computer Science* 3:e127, 2017.
  DOI 10.7717/peerj-cs.127.
* **PDF**: `Accelerating-XGBoost-with-GPU_Mitchell2017.pdf`
* **Original note (EN)**: Mitchell R, Frank E. (2017) Accelerating the
  XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127
  https://doi.org/10.7717/peerj-cs.127
* **Our view**: XGBoost's first CUDA tree builder, run entirely on the GPU,
  with exact split search via parallel prefix sums (CUB scans) over sorted
  features: interleaved scans for shallow trees, radix sort for deeper ones
  (later replaced in XGBoost by `gpu_hist`).
  3–6× over a 4-core i7 on a Titan X; trains HIGGS (10M x 28) fully in GPU
  memory. Same dataset as ours and the same exact-split, scan-based approach
  as our CUDA backend.
* **Usefulness**: ★★★★★
* **Use in paper**: Related work (GPU): the closest exact GPU tree builder.
  Method: our per-node scans are the same primitive (cite with Merrill #32).
  Results: HIGGS timing can be put in context (different hardware, boosting).

### 16. Zhang, Si, Hsieh, GPU-acceleration for Large-scale Tree Boosting

* **Citation**: H. Zhang, S. Si, C.-J. Hsieh. GPU-acceleration for Large-scale
  Tree Boosting. SysML Conference 2018. arXiv:1706.08359 (2017).
* **PDF**: `GPU-Acceleration-Large-Scale-Tree-Boosting_Zhang2018.pdf`
  (arXiv v1, was in `refs/`)
* **Original note (PL)**: Przyspieszanie budowy histogramów na GPU; ważna
  praca związana z GPU w LightGBM. Dotyczy podejścia histogramowego, a nie
  pełnego przeszukiwania wszystkich progów.
* **Our view**: Argues that histogram-based approximate splits scale better on
  GPUs than exact multi-scan or radix-sort approaches, and builds a fast
  feature-histogram kernel (LightGBM GPU). This is the opposite design choice
  to ours (exact thresholds); useful as the argument we answer: exact
  induction can be fast on GPUs too, and gives the textbook CART/C4.5 tree.
* **Usefulness**: ★★★★
* **Use in paper**: Related work (GPU): histogram vs exact. Discussion: why we
  keep exact thresholds (identical trees to the CPU, to scikit-learn and to
  C4.5; no binning parameter).

### 17. Wen et al., Efficient Gradient Boosted Decision Tree Training on GPUs

* **Citation**: Z. Wen, B. He, K. Ramamohanarao, S. Lu, J. Shi. Efficient
  Gradient Boosted Decision Tree Training on GPUs. In *IEEE IPDPS 2018*,
  pp. 234–243, 2018. DOI 10.1109/IPDPS.2018.00033.
* **PDF**: `Efficient-GBDT-Training-on-GPUs_Wen2018.pdf`
* **Original note (PL)**: Optymalizacja uczenia GBDT na GPU: kompresja,
  przydział pracy wątkom i blokom, ponowne wykorzystanie wyników. Bezpośredni
  precedens dla pracy o wydajności uczenia drzew na konferencji HPC.
* **Our view**: GPU-GBDT over sorted feature values, giving the same model
  (same RMSE) as XGBoost's exact method:
  run-length encoding to shrink sorted columns, dynamic thread/block
  allocation for irregular per-node workloads, reuse of intermediate results.
  Same problems as ours (irregular node sizes, memory traffic of sorted
  columns); a precedent for publishing tree-training performance at an HPC
  venue.
* **Usefulness**: ★★★★
* **Use in paper**: Related work (GPU); venue argument (IPDPS). Method: compare
  workload allocation for nodes of very different sizes with our size-based
  device assignment.

### 18. Wen et al., Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training (ThunderGBM)

* **Citation**: Z. Wen, J. Shi, B. He, J. Chen, K. Ramamohanarao, Q. Li.
  Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training.
  *IEEE Transactions on Parallel and Distributed Systems* 30(12):2706–2717,
  2019. DOI 10.1109/TPDS.2019.2920131.
* **PDF**: `Exploiting-GPUs-GBDT-Training-ThunderGBM_Wen2019.pdf` (author
  copy). Note: arXiv:1802.05640, which some indexes give for this paper, is a
  different paper (piece-wise linear regression trees).
* **Original note (PL)**: System ThunderGBM; optymalizacja pamięci,
  sortowania, histogramów i obsługa wielu GPU. Istotny punkt odniesienia dla
  analizy ograniczeń istniejących bibliotek.
* **Our view**: The journal version of #17: RLE compression, stable-sort-based
  partitioning of the data between children, two-stage histogram building,
  multi-GPU. Up to 10× over CPU XGBoost/LightGBM/CatBoost, and handles
  high-dimensional data where other GPU libraries fail, which is our
  ECBDL/SOREL point (cuML and XGBoost limits on 631–2381 features). Its
  stable-sort partition is the GPU counterpart of our one-pass partition.
* **Usefulness**: ★★★★
* **Use in paper**: Related work (GPU), and when discussing memory limits of
  GPU libraries on wide data. Consider ThunderGBM as an extra baseline or say
  why not (GBDT only, maintenance).

### 19. Liu, Peng, Wen, Accelerating Multi-Output GBDTs with GPUs

* **Citation**: H. Liu, X. Peng, Z. Wen. Accelerating Multi-Output GBDTs with
  GPUs. In *Proc. 54th International Conference on Parallel Processing (ICPP
  '25)*, pp. 720–729, ACM, 2025. DOI 10.1145/3754598.3754638.
* **PDF**: `Accelerating-MultiOutput-GBDTs-with-GPUs_Liu2025.pdf`
* **Original note (PL)**: Nowszy przykład: przyspieszanie GBDT na GPU,
  adaptacyjny wybór metod budowy histogramów i podział cech między GPU. Mocny
  argument za dodaniem ICPP do Waszej listy.
* **Our view**: GPU training of multi-output GBDT: adaptive choice of
  histogram construction method, memory layout for vector gradients, features
  split across GPUs; 1.7–170× speed-ups. Histogram-based and boosting, so not
  a direct comparison; shows the topic is accepted at ICPP in 2025.
* **Usefulness**: ★★★
* **Use in paper**: Related work (one sentence, recent GPU tree training);
  venue choice (ICPP). The adaptive choice of kernel by workload is a parallel
  to our choice of device by node size.

### 20. Ke et al., LightGBM

* **Citation**: G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye,
  T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In
  *Advances in Neural Information Processing Systems 30 (NIPS 2017)*,
  pp. 3146–3154, 2017. (`ke2017lightgbm`)
* **PDF**: `LightGBM-Efficient-GBDT_Ke2017.pdf`
* **Original note**: — (from `paper/references.bib`)
* **Our view**: Histogram-based GBDT with GOSS and EFB. The standard reference
  for histogram (binned) split finding, the approximation we do not make.
* **Usefulness**: ★★★
* **Use in paper**: Related work: histogram methods (with #16), and
  introduction (ensembles).

### 21. Wang et al., XGT: Fast and Secure Decision Tree Training and Inference on GPUs

* **Citation**: Q. Wang, S. Cui, L. Zhou, Y. Dong, J. Bai, Y. S. Koh,
  G. Russello. XGT: Fast and Secure Decision Tree Training and Inference on
  GPUs. *IEEE Transactions on Dependable and Secure Computing* 22(6):7479–7494,
  2025. DOI 10.1109/TDSC.2025.3597394.
* **PDF**: `XGT-Secure-DT-Training-Inference-GPUs_Wang2025.pdf` (was in
  `refs/`)
* **Original note**: — (already in `refs/`)
* **Our view**: Privacy-preserving (MPC) tree training and inference on GPUs,
  turning tree operations into matrix operations; 278× over CPU MPC. A
  different problem (secure computation dominates cost); shows GPU tree
  training is a current topic in 2025.
* **Usefulness**: ★★
* **Use in paper**: Optional, one sentence on recent GPU tree work in other
  settings.

## 4. Surveys

### 22. Costa & Pedreira, Recent advances in decision trees: an updated survey

* **Citation**: V. G. Costa, C. E. Pedreira. Recent advances in decision
  trees: an updated survey. *Artificial Intelligence Review*
  56(5):4765–4800, 2023 (online 2022). DOI 10.1007/s10462-022-10275-5.
  (Your list had "56(2):1-36"; the published issue is 56(5), pp. 4765–4800.)
* **PDF**: none (paywalled), see *Missing PDFs*.
* **Original note (EN)**: Recent advances in decision trees: an updated
  survey, Artificial Intelligence Review 56(2):1-36, Vinícius G. Costa, Carlos
  Eduardo Pedreira
* **Our view**: Survey of the last decade of decision-tree research organised
  by fitting, generalisation and interpretability. Little on parallel or GPU
  induction, so it supports the "trees are still widely used and studied"
  sentence, not the HPC part.
* **Usefulness**: ★★★
* **Use in paper**: Introduction, first paragraph (trees remain popular and
  interpretable).

### 23. Mienye & Jere, A Survey of Decision Trees: Concepts, Algorithms, and Applications

* **Citation**: I. D. Mienye, N. Jere. A Survey of Decision Trees: Concepts,
  Algorithms, and Applications. *IEEE Access* 12:86716–86727, 2024.
  DOI 10.1109/ACCESS.2024.3416838.
* **PDF**: `Survey-of-Decision-Trees-Concepts-Algorithms-Applications_Mienye2024.pdf`
  (was in `refs/`)
* **Original note (EN)**: A Survey of Decision Trees: Concepts, Algorithms, and
  Applications, I. D. Mienye and N. Jere, in IEEE Access, vol. 12,
  pp. 86716-86727, 2024, doi: 10.1109/ACCESS.2024.3416838.
* **Our view**: Recent survey with the formulas of ID3, C4.5, C5.0, CART,
  CHAID and tree ensembles, and applications (medicine, fraud). Open access.
* **Usefulness**: ★★★
* **Use in paper**: Introduction (with #22); a current reference next to
  Breiman and Quinlan for readers who want the definitions.

### 24. Rivera-Lopez et al., Induction of decision trees as classification models through metaheuristics

* **Citation**: R. Rivera-Lopez, J. Canul-Reich, E. Mezura-Montes,
  M. A. Cruz-Chávez. Induction of decision trees as classification models
  through metaheuristics. *Swarm and Evolutionary Computation* 69:101006,
  2022. DOI 10.1016/j.swevo.2021.101006.
* **PDF**: `DT-Induction-via-Metaheuristics-Survey_RiveraLopez2022.pdf` (was in
  `refs/`)
* **Original note**: — (already in `refs/`)
* **Our view**: Survey of global (evolutionary, swarm, ...) tree induction, the
  alternative to greedy top-down CART/C4.5. Lists the parallel and GPU
  versions of global induction by Jurczuk et al. (the co-author's group).
* **Usefulness**: ★★
* **Use in paper**: One sentence: greedy top-down induction remains the
  standard because global methods are far costlier; and as the entry point to
  the group's own GPU work (see the novelty section).

## 5. Algorithms, baselines and GPU primitives (from `paper/references.bib`)

### 25. Breiman et al., Classification and Regression Trees

* **Citation**: L. Breiman, J. H. Friedman, R. A. Olshen, C. J. Stone.
  *Classification and Regression Trees*. Wadsworth, 1984.
  (`breiman1984cart`)
* **PDF**: none (book).
* **Our view**: Defines what `--cart` implements: Gini, binary splits,
  cost-complexity pruning (ours at a fixed α; the book also picks α by
  cross-validation, which `./tree` leaves to the caller).
* **Usefulness**: ★★★★★
* **Use in paper**: Sect. 2.1, and the CV motivation (Breiman's K-fold CV
  needs K+1 trees).

### 26. Quinlan, C4.5: Programs for Machine Learning

* **Citation**: J. R. Quinlan. *C4.5: Programs for Machine Learning*. Morgan
  Kaufmann, 1993. (`quinlan1993c45`)
* **PDF**: none (book).
* **Our view**: Defines what `--c45` implements (gain ratio, MDL threshold
  correction from Release 8, error-based pruning).
* **Usefulness**: ★★★★★
* **Use in paper**: Sect. 2.1. Cite Quinlan 1996, "Improved use of continuous
  attributes in C4.5" (JAIR 4:77–90, open access) too if the Release 8 MDL
  correction is described.

### 27. Fayyad & Irani, On the Handling of Continuous-Valued Attributes

* **Citation**: U. M. Fayyad, K. B. Irani. On the Handling of
  Continuous-Valued Attributes in Decision Tree Generation. *Machine Learning*
  8:87–102, 1992. DOI 10.1007/BF00994007. (`fayyad1992continuous`)
* **PDF**: none (open access, but Springer blocked the download script), see
  *Missing PDFs*.
* **Our view**: Proves the best entropy threshold lies at a class boundary
  between sorted values, which lets an exact search skip candidates. Our code
  uses this boundary-point shortcut (`src/algo/split_math.h`,
  `src/build/cpu_builder.cpp`).
* **Usefulness**: ★★★★
* **Use in paper**: Method: cite where the boundary-point shortcut is
  described.

### 28. Pedregosa et al., Scikit-learn

* **Citation**: F. Pedregosa et al. Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research* 12:2825–2830, 2011.
  (`pedregosa2011sklearn`)
* **PDF**: `Scikit-learn-ML-in-Python_Pedregosa2011.pdf`
* **Our view**: CPU baseline (`DecisionTreeClassifier`, exact CART), also the
  reference for byte-identical tree checks.
* **Usefulness**: ★★★★
* **Use in paper**: Compared implementations.

### 29. Hall et al., The WEKA Data Mining Software: An Update

* **Citation**: M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann,
  I. H. Witten. The WEKA Data Mining Software: An Update. *SIGKDD
  Explorations* 11(1):10–18, 2009. DOI 10.1145/1656274.1656278.
  (`hall2009weka`)
* **PDF**: `WEKA-Data-Mining-Software-Update_Hall2009.pdf`
* **Our view**: CPU baseline (J48 = C4.5). CUDT (#2) also compares against J48.
* **Usefulness**: ★★★★
* **Use in paper**: Compared implementations.

### 30. Therneau & Atkinson, rpart

* **Citation**: T. Therneau, B. Atkinson. rpart: Recursive Partitioning and
  Regression Trees. R package, https://CRAN.R-project.org/package=rpart.
  (`therneau2023rpart`; add the version used, e.g. 4.1.x, and its year.)
* **PDF**: none (software).
* **Our view**: CPU baseline (CART in R).
* **Usefulness**: ★★★
* **Use in paper**: Compared implementations.

### 31. Raschka, Patterson, Nolet, Machine Learning in Python (cuML)

* **Citation**: S. Raschka, J. Patterson, C. Nolet. Machine Learning in
  Python: Main Developments and Technology Trends in Data Science, Machine
  Learning, and Artificial Intelligence. *Information* 11(4):193, 2020.
  DOI 10.3390/info11040193. (`raschka2020cuml`)
* **PDF**: none (open access, not downloaded), see *Missing PDFs*.
* **Our view**: The usual citation for RAPIDS cuML, our main GPU baseline.
* **Usefulness**: ★★★★
* **Use in paper**: Compared implementations.

### 32. Merrill & Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back

* **Citation**: D. Merrill, M. Garland. Single-pass Parallel Prefix Scan with
  Decoupled Look-back. NVIDIA Technical Report NVR-2016-002, 2016.
  (`merrill2016scan`)
* **PDF**: `Single-Pass-Prefix-Scan-Decoupled-Lookback_Merrill2016.pdf`
* **Our view**: The scan behind CUB; our CUDA backend's class-count prefix
  sums and partition rely on this kind of scan.
* **Usefulness**: ★★★
* **Use in paper**: Method (GPU), where scans are described.

## 6. Datasets (from `paper/references.bib`)

### 33. Baldi, Sadowski, Whiteson, HIGGS and SUSY

* **Citation**: P. Baldi, P. Sadowski, D. Whiteson. Searching for Exotic
  Particles in High-Energy Physics with Deep Learning. *Nature
  Communications* 5:4308, 2014. DOI 10.1038/ncomms5308. (`baldi2014higgs`)
* **PDF**: `HIGGS-SUSY-Exotic-Particles-Deep-Learning_Baldi2014.pdf` (arXiv
  version)
* **Our view**: Source of SUSY and HIGGS and of the "last 500,000 rows are the
  test set" rule. HIGGS is also used by Mitchell & Frank (#15).
* **Usefulness**: ★★★★
* **Use in paper**: Datasets table.

### 34. Triguero et al., ROSEFW-RF (ECBDL'14)

* **Citation**: I. Triguero, S. del Río, V. López, J. Bacardit,
  J. M. Benítez, F. Herrera. ROSEFW-RF: The winner algorithm for the ECBDL'14
  big data competition: An extremely imbalanced big data bioinformatics
  problem. *Knowledge-Based Systems* 87:69–79, 2015.
  DOI 10.1016/j.knosys.2015.05.027. (`triguero2015ecbdl`)
* **PDF**: none (open access, repository blocked the script), see *Missing
  PDFs*.
* **Our view**: Describes the ECBDL'14 protein-structure dataset (32M rows,
  631 features, 2% positives) and the winning random-forest pipeline on
  Hadoop. The bib title is shortened; use the full title above.
* **Usefulness**: ★★★★
* **Use in paper**: Datasets table; the winner needed a Hadoop cluster, our
  tree trains on one node.

### 35. Loosli, Canu, Bottou, MNIST8M

* **Citation**: G. Loosli, S. Canu, L. Bottou. Training Invariant Support
  Vector Machines Using Selective Sampling. In *Large Scale Kernel Machines*,
  pp. 301–320, MIT Press, 2007. (`loosli2007mnist8m`)
* **PDF**: none (book chapter).
* **Our view**: Origin of MNIST8M (deformed MNIST digits), also used by
  Yggdrasil (#11).
* **Usefulness**: ★★★★
* **Use in paper**: Datasets table.

### 36. Harang & Rudd, SOREL-20M

* **Citation**: R. Harang, E. M. Rudd. SOREL-20M: A Large Scale Benchmark
  Dataset for Malicious PE Detection. arXiv:2012.07634, 2020.
  (`harang2020sorel`)
* **PDF**: `SOREL-20M-Malware-Dataset_Harang2020.pdf`
* **Our view**: Source of SOREL-20M (2381 EMBER features, time-based
  train/test split), our widest dataset.
* **Usefulness**: ★★★★
* **Use in paper**: Datasets table.

---

## How the references map to the paper's story

* **Introduction, why trees and why speed** (hyperparameter tuning,
  cross-validation, ensembles): #22, #23 (trees are popular), #25 (CV needs
  K+1 trees), #14, #20 (ensembles grow thousands of trees), #1 (tree building
  inside iterative methods).
* **Large-scale exact induction**: #8, #9 (presorting), #11 (exact, deep,
  MNIST8M, high-dimensional), #7 (largest UCI sets, multicore).
* **Exact vs histogram on GPU**: exact #15, #17, #2; histogram #16, #20, #18.
  Our position: exact thresholds, identical trees on all devices.
* **CPU-GPU split**: #2 (CPU controls only), #4 (whole trees per device), #3
  (to check), #1 (size-aware batching on one device). Ours: nodes by size,
  both devices grow the tree.
* **Venues with similar papers**: CCPE (#1, #7), IPDPS (#17), TPDS (#18),
  ICPP (#19), NeurIPS (#11), FGCS (#12).

# Wielowątkowość w `TreeParallel`

`TreeParallel` używa dwóch poziomów równoległości:

- `nodeExecutor` — buduje poddrzewa równolegle (lewe dziecko w puli, prawe na bieżącym wątku).
- `featureExecutor` — w jednym węźle szuka najlepszego podziału równolegle po cechach.

Wersja jednowątkowa: `TreeSerial`.

## Dwie pule (`TaskExecutor`)

Wątek z `nodeExecutor` woła `expandOneNode` → `findBestSplitAtNode`, które korzysta z `featureExecutor`. Jedna wspólna pula mogłaby się zakleszczyć (wątek node czeka na wynik feature na tej samej kolejce). Dlatego są **osobne** executory z limitami `maxFeatureThreadCount` i `maxNodeThreadCount`.

Pliki: `task_executor.h`, `task_executor.cpp`.

## Feature — równoległość po cechach

W `findBestSplitAtNode` (`tree_parallel.cpp`):

1. Współdzielona lista wierszy (`shared_ptr`) na węzeł (tylko gdy włączona równoległość).
2. Dla każdej cechy: `featureExecutor->submit` → `evaluateFeatureSplit` (sort w `buildSortedFeatureView` + `scoreAllThresholdsForFeature`).
3. Wątek koordynujący: `future::get()` dla każdej cechy → `reduceBestSplitSearch` (wewnątrz `chooseBestSplit` po najlepszej cesze).

Próg: `minFeaturesToParallelize` (domyślnie 4, `featureCount >=` próg). Poniżej — pętla sekwencyjna po cechach.

## Node — równoległość po węzłach

Główna ścieżka: `fit()` → `buildNodeParallel(rowIndices, 0)` (synchronicznie, bez osobnego „future korzenia”).

W `buildNodeParallel`:

1. `expandOneNode` — liść albo węzeł decyzyjny + partycje wierszy.
2. Jeśli `rowCount < minRowsToParallelize` **lub** `tryStartNodeTask()` nie przejmie slotu: oba dzieci rekurencyjnie na **bieżącym** wątku (`buildNodeParallel`, nie `buildNode`).
3. W przeciwnym razie: **lewe** poddrzewo → `nodeExecutor->submit`, **prawe** → `buildNodeParallel` na bieżącym wątku, potem `leftJob.get()`.

Limit równoległych zadań node: `maxNodeTasks = max(0, maxNodeThreadCount - 1)` — zostaje co najmniej jeden worker wolny, żeby uniknąć zakleszczenia przy `future::get()`. Licznik: `activeNodeTasks` (`tryStartNodeTask` / `finishNodeTask` w jobie lewego dziecka).

## Konfiguracja

| Opcja | Znaczenie |
|-------|-----------|
| `maxFeatureThreadCount` | Wątki w `featureExecutor` (równoległe cechy w węźle) |
| `maxNodeThreadCount` | Wątki w `nodeExecutor` (równoległe poddrzewa) |
| `minFeaturesToParallelize` | Min. liczba cech, by włączyć równoległość feature |
| `minRowsToParallelize` | Min. liczba wierszy w węźle, by rozważyć równoległość node |

Domyślne wartości w `TrainingOptions` (`tree_base.h`): 4 / 4 / 4 / 32. W `main.cpp` często ustawiane są wyższe limity wątków.

## Diagram

```
fit()
  setupParallelExecutors()  → featureExecutor, nodeExecutor, maxNodeTasks
  buildNodeParallel(root)
    expandOneNode
      findBestSplitAtNode ──► featureExecutor: evaluateFeatureSplit × F
      partitionRows / partitionFromSortedView
    [dużo wierszy + wolny slot]
      nodeExecutor: buildNodeParallel(left)
      bieżący wątek: buildNodeParallel(right)
      leftJob.get()
    [mały węzeł lub brak slotu]
      buildNodeParallel(left), buildNodeParallel(right) — ten sam wątek
  fitContext_.reset()   // pule zniszczone
  finalizeFit() → prune (jeden wątek)
```

## Kompilacja

`TreeParallel` nie wymaga CUDA. Build CPU (jak zadanie „Build tree (CPU only)” w `.vscode/tasks.json`):

```bash
g++ -std=c++20 -O2 -pthread -I. \
  main.cpp tree_base.cpp tree_parallel.cpp tree_serial.cpp \
  task_executor.cpp dataset.cpp node.cpp pruning/pruning.cpp \
  -o tree_cpu
```

Pełny build z `TreeCuda` — `build_cuda.sh` / nvcc (zawiera też `tree_parallel.cpp`).

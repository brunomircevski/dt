# Wielowątkowość w `TreeParallel`

`TreeParallel` zawsze używa dwóch poziomów równoległości:

- `nodeExecutor` buduje niezależne poddrzewa równolegle.
- `attributeExecutor` szuka najlepszego podziału równolegle po cechach.

Jeśli potrzebna jest wersja jednowątkowa, użyj `TreeSerial`.

## Dwie pule (`TaskExecutor`)

Wątek z `nodeExecutor` woła `findBestSplitAtNode`, które używa `attributeExecutor`. Jedna wspólna pula mogłaby się zakleszczyć (wątek Node czeka na Attribute na tej samej kolejce). Dlatego są **osobne** executory o tym samym `maxThreadCount`.

Pliki: `task_executor.h`, `task_executor.cpp`.

## Attribute — równoległość po cechach

W `findBestSplitAtNode`:

1. Współdzielona lista wierszy (`shared_ptr`) na węzeł.
2. Dla każdej cechy: `attributeExecutor->submit` → sort + `scoreAllThresholdsForFeature`.
3. Wątek koordynujący: `future::get()` → `chooseBestSplit`.

Próg: `minFeaturesToParallelize` (domyślnie 4).

## Node — równoległość po węzłach

- `growTreeWithNodeParallelism` → `expandNodeAsync` na korzeniu.
- Dzieci lewe/prawe jako osobne joby na `nodeExecutor` (`scheduleAsyncChildren`).
- `fit()` czeka: `pendingNodeJobs == 0` i `future` korzenia.
- Mały węzeł (`rowCount < minRowsToParallelize`): fallback do `buildNode` na bieżącym wątku.

## Konfiguracja

| Opcja | Znaczenie |
|-------|-----------|
| `maxThreadCount` | Wątki w `attributeExecutor` i/lub `nodeExecutor` |
| `minFeaturesToParallelize` | Min. liczba cech dla Attribute |
| `minRowsToParallelize` | Min. liczba wierszy w węźle dla Node |

## Diagram

```
fit()
  nodeExecutor: expandNodeAsync(root)
    findBestSplitAtNode  ──► attributeExecutor: jedna cecha × F
    partitionRows
    nodeExecutor: expandNodeAsync(left), expandNodeAsync(right)
  wait: pendingNodeJobs == 0, root future
  prune (wspólne, 1 wątek)
```

## Kompilacja

```bash
g++ -std=c++17 -O2 -pthread -I. main.cpp tree_base.cpp tree_parallel.cpp tree_serial.cpp tree_cuda.cpp task_executor.cpp dataset.cpp node.cpp pruning/pruning.cpp -o tree
```

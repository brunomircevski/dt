# Wielowątkowość w `C45Tree` (GLEAMS)

Tryb wybiera się przez `TrainingOptions::gleamsMode` (`Serial`, `VDa`, `Ta`, `VDTa`). Równoległość nie włącza się już automatycznie przy `maxThreadCount > 1`.

## Tryby

| `gleamsMode` | Budowa drzewa | `findBestSplit` | Executory |
|--------------|---------------|-----------------|-----------|
| **Serial** | rekurencyjny `buildNode` (DFS) | sekwencyjnie po cechach | brak |
| **VDa** | `buildNode` (DFS) | pipeline VD (futures per cecha) | `vdExecutor` |
| **Ta** | `expandNodeAsync` (callbacki, bez czekania na dzieci) | sekwencyjnie | `taExecutor` |
| **VDTa** | `expandNodeAsync` | pipeline VD | **oba** (`vdExecutor` + `taExecutor`) |

## Dwie pule (`TaskExecutor`)

W **VDTa** worker `taExecutor` woła `findBestSplit` z pipeline VD. Jedna wspólna pula mogłaby zablokować się (worker Ta czeka na VD na tej samej kolejce). Dlatego są **osobne** executory o tym samym `maxThreadCount`.

Pliki: `task_executor.h`, `task_executor.cpp` — kontrakt `submit()` / `std::future` (docelowo ten sam interfejs pod CUDA).

## VDa — pipeline w `findBestSplit`

1. Jedna współdzielona lista wierszy (`shared_ptr`) na węzeł — bez kopiowania indeksów per cecha.
2. Dla każdej cechy: `vdExecutor->submit` → sort + `scoreAllThresholdsForFeature` w tym samym workerze.
3. Główny wątek: `future::get()` per cecha (blokada, bez busy-poll) → `chooseBestSplit`.

**Determinizm** jak w Serial (wynik per cecha zapisany po indeksie, tie-breakery po wartościach).

Próg VD: `minFeaturesToParallelize` (domyślnie 4).

## Budowa węzła — `expandOneNode`

Wspólna logika dla wszystkich trybów: liść / stop / `findBestSplit` / `partitionRows` → liść albo węzeł decyzyjny + partycje.

- **Serial / VDa:** `buildNode` → `expandOneNode`, potem rekurencyjnie `buildNode` na dzieciach.
- **Ta / VDTa:** `expandNodeAsync` → `expandOneNode`, potem `scheduleAsyncChildren` (submit na `taExecutor`).
- Małe węzły w Ta: fallback do `buildNode` (sync).

## Ta / VDTa — `expandNodeAsync`

- Zadanie węzła **nie czeka** na dzieci na puli Ta; dzieci podpinane callbackami (`pending` + mutex).
- `fit()` czeka na `pendingNodeTasks == 0` i `future` korzenia.
- Małe węzły (`rowIndices.size() < minRowsToParallelize`, domyślnie 32): `expandNodeSync` (= `buildNode`) — mniejszy narzut.
- Build release: `-O2` w `.vscode/tasks.json`.

## Konfiguracja

| Opcja | Znaczenie |
|-------|-----------|
| `gleamsMode` | Serial / VDa / Ta / VDTa |
| `maxThreadCount` | Wątki w `vdExecutor` i/lub `taExecutor` |
| `minFeaturesToParallelize` | Min. liczba cech dla VD |
| `minRowsToParallelize` | Min. liczba wierszy w węźle dla Ta |

## Diagram (VDTa)

```
fit()
  taExecutor: expandNodeAsync(root)
    findBestSplit()  ──► vdExecutor: buildSortedFeatureView × F
                       koordynator: score gdy future ready
    partitionRows (1×)
    taExecutor: expandNodeAsync(left), expandNodeAsync(right)  // bez latch na dzieciach
  wait: pendingNodeTasks == 0, root future
  pruning (1 wątek)
```

## Weryfikacja

Ten sam dataset i opcje CART — dla wszystkich czterech `gleamsMode` oczekiwana **identyczna** głębokość, liczba węzłów i accuracy; różny czas budowy.

Przykład (Covertype, `maxDepth = 5`, `maxThreadCount = 28`):

```bash
g++ -std=c++17 -O2 -pthread -I. main.cpp c45_tree.cpp task_executor.cpp dataset.cpp node.cpp tree_visualization.cpp pruning/*.cpp -o tree
./tree   # ustaw gleamsMode w main.cpp lub benchmark
```

## Wyniki testów (czas budowy)

Konfiguracja wspólna: CART (Gini, `MaxGain`), `maxThreadCount = 28`, `-O2`, `minFeaturesToParallelize = 4`, `minRowsToParallelize = 32`. Metryka: `build time` [s].

| Dataset | `maxDepth` | Serial [s] | Ta [s] | VDa [s] | VDTa [s] |
|---------|------------|------------|--------|---------|----------|
| `covertype_10x_smaller.csv` | 35 | 9.0 | 3.8 | 1.8 | 1.4 |
| `covertype.csv` | 5 | 38.8 | 32.3 | 10.2 | 10.3 |
| `covertype.csv` | 46 | 110.9 | 45.7 | 22.8 | 20.8 |

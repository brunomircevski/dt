# Wielowątkowość w `C45Tree`

## Stan aktualny: **VD-GLEAMS** (Vertical Data Parallelism)

Zrównoleglenie **po cechach (kolumnach)** przy wyszukiwaniu splitu w jednym węźle. Budowa drzewa (`buildNode`), ocena progów i pruning pozostają sekwencyjne.

## Jak to działa

W każdym węźle `findBestSplit` musi ocenić wszystkie cechy. Dla każdej cechy `buildSortedFeatureView` sortuje wiersze węzła i buduje listę progów (koszt ~O(n log n)) — to dominująca praca.

**VD:** te sortowania uruchamiane są równolegle — `SplitThreadPool::parallel_for` po `featureIndex`, każdy wątek zapisuje do `featureViews[i]`. Po zakończeniu (wywołujący czeka) sekwencyjnie oceniane są progi (`scoreSplitFromSorted`), wybierany jest najlepszy split, a `buildNode` rekurencyjnie idzie w lewo/prawo jednym wątkiem.

```
fit()
  buildNode()                         ← 1 wątek (DFS)
    findBestSplit()
      parallel: buildSortedFeatureView × F   ← VD (pula wątków)
      sekwencyjnie: scoreSplitFromSorted, wybór splitu
    partitionRows (1×)
  pruning                             ← 1 wątek
```

Pula `SplitThreadPool` żyje przez całe `fit()` (`c45_tree.cpp`). Równoległość na progach była wcześniej testowana i **usunięta** — po optymalizacji prefix sums nie dawała zysku.

---

## Konfiguracja

| Opcja | Znaczenie |
|-------|-----------|
| `maxThreadCount` | Liczba wątków (1 = bez równoległości) |
| `minFeaturesToParallelize` | Min. liczba cech, by włączyć VD (domyślnie 4) |

Ustawienia w `c45_tree.h`, `main.cpp` (np. 28 wątków, próg 3 cech).

---

## Wyniki (Covertype, 58 101 próbek, CART, `maxDepth = 5`)

| Wątki | Czas budowy | Model |
|-------|-------------|--------|
| 1 | ~22 s | depth 5, 45 węzłów, acc. 71,05% |
| 28 | ~3,8 s | **identyczny** |

Przyspieszenie ~5,8× przy tej samej strukturze drzewa. Sort `(value, label)` zapewnia determinizm przy remisach wartości.

Optymalizacja algorytmu (bez wątków): `SortedFeatureView` + prefix sums — sort raz na cechę, `partitionRows` tylko po wybranym splicie (~38 s → ~22 s na 1 wątku).

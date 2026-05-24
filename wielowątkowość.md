# Optymalizacja i wielowątkowość — podsumowanie

Dokument opisuje zmiany w budowie drzewa decyzyjnego (`C45Tree`): optymalizację wyszukiwania splitów, ewolucję równoległości oraz wyniki pomiarów na Covertype (58 101 próbek, konfiguracja CART z `main.cpp`).

## Konfiguracja pomiarów

- **Dataset:** `covertype` (58 101 próbek)
- **CART:** Gini, `MaxGain`, `CostComplexity`, `ccpAlpha = 10`, `maxDepth = 5`
- **Sortowanie:** `(value, label)` — deterministyczna kolejność jak w pierwotnej implementacji
- **Metryki modelu (identyczne dla 1 i 28 wątków):** depth 5, 45 węzłów, accuracy 71,0521%

| `maxThreadCount` | Build time | Prune time |
|------------------|------------|------------|
| 1 | 22 087 ms (~22,1 s) | 54,8 ms |
| 28 | 3 791 ms (~3,8 s) | 56,2 ms |

Przyspieszenie budowy przy 28 wątkach: **~5,8×** względem wersji jednowątkowej (ta sama dokładność i struktura drzewa).

Dla porównania historycznego — **przed optymalizacją sortowania** (stary kod, wielokrotne `partitionRows` na kandydatów, równoległość na progach):

- Build ~38 589 ms (~38,6 s), CPU często ~100% podczas oceny splitów
- Ten sam model po przywróceniu sortu `(value, label)` co obecnie przy `maxThreadCount = 1`

---

## 1. Optymalizacja algorytmu (bez wątków)

### Problem

W `findBestSplit` dla każdego kandydata `(cecha, próg)` wywoływano `scoreSplit` → `partitionRows`, czyli **pełny skan wierszy w węźle**. Przy wielu progach koszt był rzędu **O(n × liczba progów)** na cechę i węzeł.

Sortowanie odbywało się raz na cechę w `collectNumericThresholdCandidates`, ale **kolejność sortowania była wyrzucana** — przy każdym progu dane dzielono od zera.

### Rozwiązanie: `SortedFeatureView` + prefix sums

- **`buildSortedFeatureView`** — raz na cechę w węźle: sort, lista progów, `leftSizes`, `prefixClassCounts`
- **`scoreSplitFromSorted`** — ocena progu z histogramów (bez `partitionRows`)
- **`impurityFromClassCounts`** — entropia/Gini z mapy liczników klas
- **`partitionRows`** — tylko **raz** po wybraniu zwycięskiego splitu w `buildNode`

### Efekt

- Znacznie krótszy czas budowy (~38 s → ~22 s przy jednym wątku)
- Niższe średnie obciążenie CPU (krótsze, lżejsze fazy równoległe; większy udział sekwencyjnego sortowania)

### Sortowanie — ważna uwaga

| Wariant sortu | Wynik |
|---------------|--------|
| Tylko `value` | Szybszy zapis (~19,8 s), **inne drzewo** (np. 53 węzły, 71,16% na train) |
| `value`, potem `label` | Zgodne z pierwotnym kodem (45 węzłów, 71,05%) — **wybrane** ze względu na determinizm |

Przy powtarzających się wartościach cechy i różnych etykietach kolejność sąsiadów wpływa na listę progów. To nie błąd Gini/entropii, tylko inna definicja kandydatów.

---

## 2. Wielowątkowość — co próbowano

### 2.1. Pula wątków `SplitThreadPool` + `parallel_for`

Uniwersalny wzorzec:

```cpp
parallel_for(count, [&](std::size_t i) { /* praca dla i */ });
```

- Wątki tworzone na czas `fit()`, niszczone po treningu
- Wywołujący **czeka** do końca (jak `Task.WhenAll`)
- Każdy indeks `i` powinien pisać w osobne miejsce (brak współdzielonej mutacji)

### 2.2. Równoległość na ocenie progów — **usunięta**

**Gdzie:** `findBestSplit` — równoległe `scoreSplit` / `scoreSplitFromSorted` dla listy kandydatów.

**Dlaczego kiedyś miała sens:** przy starym `partitionRows` na każdy próg praca była ciężka → dobre nasycenie CPU (~100%).

**Dlaczego usunięto:** po optymalizacji prefix sums ocena jednego progu jest bardzo tania. Pomiary na pełnym Covertype:

| Wątki | Build (po optymalizacji, przed przeniesieniem równoległości) |
|-------|----------------------------------------------------------------|
| 1 | ~22 535 ms |
| 28 | ~22 284 ms |

Różnica w granicy szumu; narzut puli ≈ zysk. **Nie dawała praktycznego przyspieszenia.**

### 2.3. Równoległość na sortowaniu cech — **obecne rozwiązanie**

**Gdzie:** `findBestSplit` — `parallel_for` po `featureIndex` wywołujący `buildSortedFeatureView`.

**Dlaczego ma sens:** `buildSortedFeatureView` woływany jest **w każdym węźle** z szukaniem splitu, **raz na każdą kolumnę** — dominujący koszt to sort O(n log n) na dużych węzłach. Cechy w jednym węźle są niezależne.

**Konfiguracja:**

- `maxThreadCount` — liczba wątków puli (1 = sekwencyjnie)
- `minFeaturesToParallelize` (domyślnie 4) — minimalna liczba cech, by włączyć równoległość

**Wynik:** ~22 s → ~3,8 s przy 28 wątkach (ten sam model).

### 2.4. Co jeszcze nie zaimplementowano (kolejne kroki)

| Pomysł | Potencjał | Uwagi |
|--------|-----------|--------|
| Równoległe `buildNode` (lewa / prawa gałąź) | średni | tylko 2 zadania na poziom — mały narzut puli |
| Kolejka węzłów zamiast rekurencji DFS | wysoki | większa przebudowa |
| Równoległość w przycinaniu | niski | krótki etap (~55 ms) |

---

## 3. Przepływ czasu CPU (intuicja)

```
fit()
  buildNode()                    ← głównie 1 wątek (rekurencja)
    findBestSplit()
      parallel: buildSortedFeatureView × F   ← 28 wątków (krótkie „piki” CPU)
      sekwencyjnie: scoreSplitFromSorted    ← tanie
      reduceBestPerFeature / chooseBestSplit
    partitionRows (1×)           ← 1 wątek
  applySelectedPruning()         ← 1 wątek
```

Po optymalizacji niskie średnie CPU nie oznacza wolniejszego programu — oznacza **mniej pracy** i **krótsze** epizody równoległości. Pełne ~100% CPU wraca sensownie dopiero tam, gdzie równoleglimy sorty po cechach.

---

## 4. Pliki i symbole

| Element | Plik |
|---------|------|
| `SplitThreadPool`, `parallel_for` | `c45_tree.cpp` |
| `SortedFeatureView`, `buildSortedFeatureView`, `scoreSplitFromSorted` | `c45_tree.cpp`, `c45_tree.h` |
| Opcje `maxThreadCount`, `minFeaturesToParallelize` | `c45_tree.h`, `main.cpp` |
| Wizualizacja opcji w SVG | `tree_visualization.cpp` |

Usunięte / zastąpione: `collectNumericThresholdCandidates`, `scoreSplit` z pętlą `partitionRows`, `minCandidatesToParallelize`, `shouldParallelizeSplitSearch`.

---

## 5. Wnioski praktyczne

1. **Największy zysk:** sort raz na cechę + prefix sums (algorytm), potem równoległość po cechach (wątki).
2. **Równoległość na progach** była sensowna tylko przy starym, drogim `partitionRows` — po refaktorze zbędna.
3. **Determinizm:** sort `(value, label)` — powtarzalne drzewo niezależnie od kolejności wierszy w CSV przy remisach wartości.
4. **Produkcja:** `maxThreadCount` ≈ liczba rdzeni; dla małych datasetów z małą liczbą cech można zostawić `1` (próg `minFeaturesToParallelize`).

#pragma once

#include "dataset.h"
#include "node.h"

#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

enum class ImpurityMeasure {
  // Classic C4.5 uses entropy.
  Entropy,
  // Gini is kept only as an experiment/comparison mode.
  Gini
};

enum class SplitSelectionMode {
  // Classic C4.5: uses information gain and gain ratio.
  ClassicC45,
  // C4.5 variant: filters candidates with below-average gain.
  MeanGainFiltered,
  // CART style: purely maximizes information/Gini gain.
  MaxGain
};

enum class PruningMode {
  // Bez przycinania: zostawia całe drzewo jak urosło.
  None,
  // Zamienia poddrzewo na liść klasy większościowej,
  // jeśli pessymistyczny błąd liścia nie jest gorszy niż poddrzewa.
  PessimisticErrorPrune,
  // Pessymistyczne przycinanie bardziej w stylu C4.5:
  // porównuje liść vs sumę błędów wszystkich liści w poddrzewie.
  C45PessimisticPrune,
  // CART: Minimal Cost-Complexity Pruning (używa ccpAlpha).
  CostComplexityPrune,
  // Lightweight: Reduced Error Pruning (używa trafności
  // treningowej/walidacyjnej).
  ReducedErrorPrune
};

struct TrainingOptions {
  // KONFIGURACJA DLA ALGORYTMÓW:
  // CART: impurityMeasure=Gini, splitSelectionMode=MaxGain,
  // pruningMode=CostComplexityPrune C4.5: impurityMeasure=Entropy,
  // splitSelectionMode=ClassicC45, pruningMode=C45PessimisticPrune

  // Maksymalna głębokość drzewa. Zapobiega zbytniemu skomplikowaniu modelu.
  int maxDepth = -1;

  // Minimalna liczba próbek w węźle, aby podjąć próbę podziału.
  // Chroni przed dzieleniem małych grup (szumu).
  std::size_t minSamplesToSplit = 2;

  // Minimalna liczba próbek wymagana w każdym nowym liściu po podziale.
  // Gwarantuje statystyczną istotność liści.
  std::size_t minSamplesPerLeaf = 1;

  // Wybiera metodę upraszczania (przycinania) drzewa po fazie wzrostu.
  PruningMode pruningMode = PruningMode::None;

  // Kryterium wyboru najlepszego podziału spośród kandydatów.
  SplitSelectionMode splitSelectionMode = SplitSelectionMode::ClassicC45;

  // Tolerancja dla porównań zmiennoprzecinkowych (double).
  // Zapobiega tworzeniu sztucznych progów dla prawie identycznych wartości.
  double epsilon = 1e-9;

  // Współczynnik ufności dla pesymistycznego przycinania C4.5 (PEP).
  // Mniejsze wartości = silniejsze, bardziej agresywne przycinanie. Przykład:
  // 0.1.
  double pruningConfidenceFactor = 0.25;

  // Parametr złożoności dla przycinania Cost-Complexity (CART).
  // Kara za każdy dodatkowy liść; większa wartość = mniejsze drzewo. Przykład:
  // 0.01.
  double ccpAlpha = 0.01;

  // Metoda mierzenia "nieczystości" węzła (Entropia dla C4.5, Gini dla CART).
  // Podstawa oceny jakości podziału. Przykład: impurityMeasure =
  // ImpurityMeasure::Gini.
  ImpurityMeasure impurityMeasure = ImpurityMeasure::Entropy;
};

struct SplitResult {
  // If false, we did not find any useful split.
  bool valid = false;

  // Which feature gave the best split.
  std::size_t featureIndex = 0;
  std::string featureName;

  // Threshold for the question:
  // "Is feature <= threshold?"
  double threshold = 0.0;

  // These values are stored mainly for learning/debugging.
  // They let us print how good the chosen split was.
  double informationGain = 0.0;
  double splitInformation = 0.0;
  double gainRatio = 0.0;
};

class C45Tree {
public:
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{});
  std::string predict(const Sample &sample) const;
  void print(std::ostream &output) const;
  int treeDepth() const;
  std::size_t nodeCount() const;

  // The next functions are public so they can be explored from main()
  // and studied separately.
  double entropy(const std::vector<std::size_t> &rowIndices) const;
  double giniIndex(const std::vector<std::size_t> &rowIndices) const;
  double impurity(const std::vector<std::size_t> &rowIndices) const;
  double informationGain(
      const std::vector<std::size_t> &rowIndices,
      const std::vector<std::vector<std::size_t>> &partitions) const;
  double splitInformation(
      const std::vector<std::size_t> &rowIndices,
      const std::vector<std::vector<std::size_t>> &partitions) const;
  SplitResult findBestSplit(const std::vector<std::size_t> &rowIndices) const;

private:
  struct PartitionedRows {
    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;
  };

  // We only keep a pointer to the dataset. The tree does not own the dataset.
  // "const" means the tree promises not to modify it.
  const Dataset *dataset_ = nullptr;

  // root_ points to the first node in the tree.
  std::unique_ptr<Node> root_;

  // The last training call stores its configuration here so all helper
  // functions can use the same settings while the tree is being built.
  TrainingOptions options_;

  std::map<std::string, int>
  computeClassCounts(const std::vector<std::size_t> &rowIndices) const;
  std::unique_ptr<Node> buildNode(const std::vector<std::size_t> &rowIndices,
                                  int depth) const;
  std::vector<double>
  collectNumericThresholdCandidates(const std::vector<std::size_t> &rowIndices,
                                    std::size_t featureIndex) const;
  SplitResult scoreSplit(const std::vector<std::size_t> &rowIndices,
                         std::size_t featureIndex, double threshold) const;
  SplitResult chooseBestSplit(const std::vector<SplitResult> &candidates) const;
  bool shouldStopGrowing(const std::vector<std::size_t> &rowIndices,
                         int depth) const;
  PartitionedRows partitionRows(const std::vector<std::size_t> &rowIndices,
                                std::size_t featureIndex,
                                double threshold) const;
  void applySelectedPruning(const std::vector<std::size_t> &rowIndices);
  void pruneWithPessimisticError(std::unique_ptr<Node> &node,
                                 const std::vector<std::size_t> &rowIndices);
  void pruneWithC45Pessimistic(std::unique_ptr<Node> &node,
                               const std::vector<std::size_t> &rowIndices);
  void pruneWithCostComplexity(std::unique_ptr<Node> &node,
                               const std::vector<std::size_t> &rowIndices);
  void pruneWithReducedError(std::unique_ptr<Node> &node,
                             const std::vector<std::size_t> &rowIndices);
  double estimatePessimisticLeafErrorCount(std::size_t observedErrors,
                                           std::size_t sampleCount) const;
  double estimateSubtreePessimisticErrorCount(
      const Node *node, const std::vector<std::size_t> &rowIndices) const;
  std::size_t
  countCorrectPredictions(const Node *node,
                          const std::vector<std::size_t> &rowIndices) const;
  std::size_t countLeafNodes(const Node *node) const;
  std::size_t countNodes(const Node *node) const;
  int computeTreeDepth(const Node *node) const;
  bool allSameLabel(const std::vector<std::size_t> &rowIndices) const;
  std::string
  getMajorityLabel(const std::vector<std::size_t> &rowIndices) const;
  void printNode(const Node *node, std::ostream &output, int depth,
                 const std::string &edgeText) const;
};

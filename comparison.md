## Python `scikit-learn` (`DecisionTreeClassifier`) (Weka J48 - podobnie)

```text

Learned tree:
ROOT: if PetalWidthCm <= 0.800
  yes: Leaf -> Iris-setosa
  no: if PetalWidthCm <= 1.750
    yes: if PetalLengthCm <= 4.950
      yes: if PetalWidthCm <= 1.650
        yes: Leaf -> Iris-versicolor
        no: Leaf -> Iris-virginica
      no: if PetalWidthCm <= 1.550
        yes: Leaf -> Iris-virginica
        no: if PetalLengthCm <= 5.450
          yes: Leaf -> Iris-versicolor
          no: Leaf -> Iris-virginica
    no: if PetalLengthCm <= 4.850
      yes: if SepalWidthCm <= 3.100
        yes: Leaf -> Iris-virginica
        no: Leaf -> Iris-versicolor
      no: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 150
  accuracy = 100.0000%
```

## Python `c4dot5-decision-tree`

```text

Learned tree:
ROOT: if PetalLengthCm <= 1.900
  yes: Leaf -> Iris-setosa
  no: if PetalWidthCm <= 1.700
    yes: if PetalLengthCm <= 5.300
      yes: if PetalLengthCm <= 4.900
        yes: if PetalWidthCm <= 1.600
          yes: Leaf -> Iris-versicolor
          no: Leaf -> Iris-virginica
        no: if PetalWidthCm <= 1.500
          yes: Leaf -> Iris-virginica
          no: Leaf -> Iris-versicolor
      no: Leaf -> Iris-virginica
    no: if PetalLengthCm <= 4.800
      yes: if SepalLengthCm <= 5.900
        yes: Leaf -> Iris-versicolor
        no: Leaf -> Iris-virginica
      no: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 150
  accuracy = 100.0000%
```

## My first implementation C++

```text

Learned tree:
ROOT: if PetalLengthCm <= 2.450
  yes: Leaf -> Iris-setosa
  no: if PetalWidthCm <= 1.750
    yes: if SepalLengthCm <= 7.100
      yes: if PetalLengthCm <= 5.050
        yes: if SepalLengthCm <= 4.950
          yes: if SepalWidthCm <= 2.450
            yes: Leaf -> Iris-versicolor
            no: Leaf -> Iris-virginica
          no: if SepalWidthCm <= 2.250
            yes: if PetalLengthCm <= 4.750
              yes: Leaf -> Iris-versicolor
              no: Leaf -> Iris-virginica
            no: Leaf -> Iris-versicolor
        no: if SepalLengthCm <= 6.050
          yes: Leaf -> Iris-versicolor
          no: Leaf -> Iris-virginica
      no: Leaf -> Iris-virginica
    no: if SepalWidthCm <= 3.150
      yes: Leaf -> Iris-virginica
      no: if SepalLengthCm <= 6.050
        yes: Leaf -> Iris-versicolor
        no: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 150
  accuracy = 100.0000%
```

## MinSamplesPerLeaf = 2

```
ROOT [n=150]: if PetalLengthCm <= 2.450
  yes [n=50]: Leaf -> Iris-setosa
  no [n=100]: if PetalWidthCm <= 1.750
    yes [n=54]: if PetalLengthCm <= 5.050
      yes [n=50]: if SepalLengthCm <= 4.950
        yes [n=2]: Leaf -> Iris-versicolor
        no [n=48]: if SepalWidthCm <= 2.250
          yes [n=4]: Leaf -> Iris-versicolor
          no [n=44]: Leaf -> Iris-versicolor
      no [n=4]: if SepalWidthCm <= 2.750
        yes [n=2]: Leaf -> Iris-versicolor
        no [n=2]: Leaf -> Iris-virginica
    no [n=46]: if SepalWidthCm <= 3.150
      yes [n=32]: Leaf -> Iris-virginica
      no [n=14]: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 146
  accuracy = 97.3333%
```

## MinSamplesPerLeaf = 2 + TrainingAccuracyPrune post pruning
## MinSamplesPerLeaf = 2 + PessimisticErrorPrune post pruning

```
Learned tree:
ROOT [n=150]: if PetalLengthCm <= 2.450
  yes [n=50]: Leaf -> Iris-setosa
  no [n=100]: if PetalWidthCm <= 1.750
    yes [n=54]: if PetalLengthCm <= 5.050
      yes [n=50]: Leaf -> Iris-versicolor
      no [n=4]: Leaf -> Iris-virginica
    no [n=46]: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 146
  accuracy = 97.3333%
```

# Python `c4dot5-decision-tree` domyślnie - takie samo

```
ROOT: if PetalLengthCm <= 1.900
  yes: Leaf -> Iris-setosa
  no: if PetalWidthCm <= 1.700
    yes: if PetalLengthCm <= 5.300
      yes: Leaf -> Iris-versicolor
      no: Leaf -> Iris-virginica
    no: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 146
  accuracy = 97.3333%
```

## Wnioski
- TrainingAccuracyPrune Post pruning nie pogarsza wyniku, a upraszcza drzewo
- PessimisticErrorPrune Post pruning lekko pogarsza wynik, a znacznie upraszcza drzewo
- Zwiększanie MinSamplesPerLeaf likwiduje szum, upraszcza drzewo
- Entropy i Gini tworzą te same drzewo
- Wyniki nadal różne niż sckit-learning i weka
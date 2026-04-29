## Python `scikit-learn` (`DecisionTreeClassifier`)

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

## Python `c4dot5-decision-tree` (default)

```text

Learned tree:
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

## Python `c4dot5-decision-tree` (relaxed stop rule)

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

## My implementation C++

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

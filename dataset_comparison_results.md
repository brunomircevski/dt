# Dataset Comparison Results

## `iris.csv`

### Python sklearn
```text
ROOT [n=150]: if PetalWidthCm <= 0.800
  yes [n=50]: Leaf -> Iris-setosa
  no [n=100]: if PetalWidthCm <= 1.750
    yes [n=54]: if PetalLengthCm <= 4.950
      yes [n=48]: if PetalWidthCm <= 1.650
        yes [n=47]: Leaf -> Iris-versicolor
        no [n=1]: Leaf -> Iris-virginica
      no [n=6]: Leaf -> Iris-virginica
    no [n=46]: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 147
  accuracy = 98.0000%
```

### Python c4dot5
```text
ROOT [n=150]: if PetalLengthCm <= 1.900
  yes [n=50]: Leaf -> Iris-setosa
  no [n=100]: if PetalWidthCm <= 1.700
    yes [n=54]: if PetalLengthCm <= 5.300
      yes [n=52]: Leaf -> Iris-versicolor
      no [n=2]: Leaf -> Iris-virginica
    no [n=46]: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 146
  accuracy = 97.3333%
```

### C++ implementation
```text
ROOT [n=150]: if PetalLengthCm <= 2.450
  yes [n=50]: Leaf -> Iris-setosa
  no [n=100]: if PetalWidthCm <= 1.750
    yes [n=54]: if SepalLengthCm <= 7.100
      yes [n=53]: if PetalLengthCm <= 5.050
        yes [n=50]: Leaf -> Iris-versicolor
        no [n=3]: if SepalLengthCm <= 6.050
          yes [n=1]: Leaf -> Iris-versicolor
          no [n=2]: Leaf -> Iris-virginica
      no [n=1]: Leaf -> Iris-virginica
    no [n=46]: Leaf -> Iris-virginica

Prediction summary:
  checked samples = 150
  correct predictions = 147
  accuracy = 98.0000%
```

## `diabetes.csv`

### Python sklearn
```text
ROOT [n=768]: if Glucose <= 127.500
  yes [n=485]: Leaf -> 0
  no [n=283]: Leaf -> 1


Prediction summary:
  checked samples = 768
  correct predictions = 565
  accuracy = 73.5677%
```

### Python c4dot5
```text
ROOT [n=768]: if Glucose <= 166.000
  yes [n=689]: if Glucose <= 154.000
    yes [n=646]: Leaf -> 0
    no [n=43]: if Insulin <= 540.000
      yes [n=41]: if DiabetesPedigreeFunction <= 0.141
        yes [n=2]: Leaf -> 0
        no [n=39]: if Age <= 53.000
          yes [n=34]: if SkinThickness <= 42.000
            yes [n=31]: Leaf -> 1
            no [n=3]: if Glucose <= 163.000
              yes [n=1]: Leaf -> 1
              no [n=2]: Leaf -> 0
          no [n=5]: if BloodPressure <= 86.000
            yes [n=4]: Leaf -> 0
            no [n=1]: Leaf -> 1
      no [n=2]: Leaf -> 0
  no [n=79]: if BMI <= 23.100
    yes [n=1]: Leaf -> 0
    no [n=78]: if Age <= 66.000
      yes [n=77]: Leaf -> 1
      no [n=1]: Leaf -> 0

Prediction summary:
  checked samples = 768
  correct predictions = 586
  accuracy = 76.3021%
```

### C++ implementation
```text
ROOT [n=768]: if Glucose <= 165.500
  yes [n=686]: if Glucose <= 154.500
    yes [n=646]: if BMI <= 45.400
      yes [n=625]: if BMI <= 26.450
        yes [n=156]: if BMI <= 9.100
          yes [n=11]: if Pregnancies <= 7.500
            yes [n=9]: Leaf -> 0
            no [n=2]: Leaf -> 1
          no [n=145]: Leaf -> 0
        no [n=469]: if Glucose <= 92.500
          yes [n=95]: if Insulin <= 234.000
            yes [n=94]: if Glucose <= 28.500
              yes [n=4]: if Pregnancies <= 3.000
                yes [n=2]: Leaf -> 0
                no [n=2]: Leaf -> 1
              no [n=90]: if DiabetesPedigreeFunction <= 1.096
                yes [n=86]: Leaf -> 0
                no [n=4]: if Pregnancies <= 2.500
                  yes [n=2]: Leaf -> 0
                  no [n=2]: Leaf -> 1
            no [n=1]: Leaf -> 1
          no [n=374]: if BMI <= 44.350
            yes [n=368]: if SkinThickness <= 49.500
              yes [n=363]: if BloodPressure <= 15.000
                yes [n=14]: if DiabetesPedigreeFunction <= 0.138
                  yes [n=1]: Leaf -> 0
                  no [n=13]: if Age <= 39.500
                    yes [n=10]: Leaf -> 1
                    no [n=3]: Leaf -> 0
                no [n=349]: if Age <= 28.500
                  yes [n=150]: if Pregnancies <= 7.000
                    yes [n=149]: Leaf -> 0
                    no [n=1]: Leaf -> 1
                  no [n=199]: if BloodPressure <= 104.000
                    yes [n=197]: Leaf -> 1
                    no [n=2]: Leaf -> 0
              no [n=5]: Leaf -> 0
            no [n=6]: Leaf -> 0
      no [n=21]: if Glucose <= 84.500
        yes [n=2]: Leaf -> 0
        no [n=19]: if Glucose <= 153.000
          yes [n=18]: if SkinThickness <= 55.500
            yes [n=17]: if DiabetesPedigreeFunction <= 0.127
              yes [n=1]: Leaf -> 0
              no [n=16]: if Age <= 22.500
                yes [n=1]: Leaf -> 0
                no [n=15]: Leaf -> 1
            no [n=1]: Leaf -> 0
          no [n=1]: Leaf -> 0
    no [n=40]: if Insulin <= 542.500
      yes [n=38]: if DiabetesPedigreeFunction <= 0.141
        yes [n=2]: Leaf -> 0
        no [n=36]: if BloodPressure <= 51.000
          yes [n=1]: Leaf -> 0
          no [n=35]: if Glucose <= 164.500
            yes [n=32]: if Age <= 53.500
              yes [n=29]: if SkinThickness <= 42.000
                yes [n=27]: if BloodPressure <= 65.000
                  yes [n=7]: if BloodPressure <= 63.000
                    yes [n=5]: Leaf -> 1
                    no [n=2]: Leaf -> 0
                  no [n=20]: Leaf -> 1
                no [n=2]: if Pregnancies <= 0.500
                  yes [n=1]: Leaf -> 1
                  no [n=1]: Leaf -> 0
              no [n=3]: if Pregnancies <= 3.500
                yes [n=1]: Leaf -> 1
                no [n=2]: Leaf -> 0
            no [n=3]: if Pregnancies <= 7.500
              yes [n=2]: Leaf -> 0
              no [n=1]: Leaf -> 1
      no [n=2]: Leaf -> 0
  no [n=82]: if BMI <= 23.100
    yes [n=1]: Leaf -> 0
    no [n=81]: if Age <= 66.500
      yes [n=80]: if BloodPressure <= 93.000
        yes [n=74]: Leaf -> 1
        no [n=6]: if Pregnancies <= 7.500
          yes [n=4]: if Pregnancies <= 2.000
            yes [n=1]: Leaf -> 1
            no [n=3]: Leaf -> 0
          no [n=2]: Leaf -> 1
      no [n=1]: Leaf -> 0

Prediction summary:
  checked samples = 768
  correct predictions = 627
  accuracy = 81.6406%
```

## `penguins.csv`

### Python sklearn
```text
ROOT [n=342]: if flipper_length_mm <= 206.500
  yes [n=213]: if bill_length_mm <= 43.350
    yes [n=150]: Leaf -> Adelie
    no [n=63]: Leaf -> Chinstrap
  no [n=129]: if bill_depth_mm <= 17.650
    yes [n=122]: Leaf -> Gentoo
    no [n=7]: Leaf -> Chinstrap

Prediction summary:
  checked samples = 342
  correct predictions = 330
  accuracy = 96.4912%
```

### Python c4dot5
```text
ROOT [n=342]: if flipper_length_mm <= 206.000
  yes [n=213]: if bill_length_mm <= 44.500
    yes [n=152]: Leaf -> Adelie
    no [n=61]: if bill_depth_mm <= 15.300
      yes [n=1]: Leaf -> Gentoo
      no [n=60]: Leaf -> Chinstrap
  no [n=129]: if bill_depth_mm <= 18.900
    yes [n=124]: if bill_depth_mm <= 17.600
      yes [n=122]: Leaf -> Gentoo
      no [n=2]: Leaf -> Adelie
    no [n=5]: Leaf -> Chinstrap

Prediction summary:
  checked samples = 342
  correct predictions = 333
  accuracy = 97.3684%
```

### C++ implementation
```text
ROOT [n=342]: if flipper_length_mm <= 207.500
  yes [n=215]: if bill_length_mm <= 44.600
    yes [n=152]: if bill_length_mm <= 42.350
      yes [n=139]: Leaf -> Adelie
      no [n=13]: if bill_depth_mm <= 17.450
        yes [n=4]: Leaf -> Chinstrap
        no [n=9]: if flipper_length_mm <= 199.500
          yes [n=8]: Leaf -> Adelie
          no [n=1]: Leaf -> Chinstrap
    no [n=63]: if bill_depth_mm <= 15.450
      yes [n=2]: Leaf -> Gentoo
      no [n=61]: if bill_depth_mm <= 21.150
        yes [n=60]: if body_mass_g <= 4575.000
          yes [n=59]: Leaf -> Chinstrap
          no [n=1]: Leaf -> Adelie
        no [n=1]: Leaf -> Adelie
  no [n=127]: if bill_depth_mm <= 17.650
    yes [n=121]: Leaf -> Gentoo
    no [n=6]: if bill_length_mm <= 46.550
      yes [n=2]: Leaf -> Adelie
      no [n=4]: Leaf -> Chinstrap

Prediction summary:
  checked samples = 342
  correct predictions = 340
  accuracy = 99.4152%
```


# Po zmianach
```
Classic

Enthropy:
  checked samples = 768
  correct predictions = 768
  tree depth = 76
  node count = 361
  accuracy = 100.0000%
  
Gini:
  checked samples = 768
  correct predictions = 768
  tree depth = 81
  node count = 387
  accuracy = 100.0000%
  
Gini + PessimisticErrorPrune:
  checked samples = 768
  correct predictions = 748
  tree depth = 81
  node count = 305
  accuracy = 97.3958%
  
Enthropy + PessimisticErrorPrune:
  checked samples = 768
  correct predictions = 747
  tree depth = 76
  node count = 275
  accuracy = 97.2656%
```

```
Custom MeanGainFiltered

Enthorpy:
  checked samples = 768
  correct predictions = 768
  tree depth = 55
  node count = 363
  accuracy = 100.0000%
  
Gini:
  checked samples = 768
  correct predictions = 768
  tree depth = 46
  node count = 361
  accuracy = 100.0000%
  
Gini + PessimisticErrorPrune:
  checked samples = 768
  correct predictions = 747
  tree depth = 46
  node count = 275
  accuracy = 97.2656%
  
Enthropy + PessimisticErrorPrune
  checked samples = 768
  correct predictions = 751
  tree depth = 55
  node count = 293
  accuracy = 97.7865%
```

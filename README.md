# Naive Bayes Classifier (Iris Dataset)

This project implements a **nominal (categorical) Naive Bayes classifier** in C++ from scratch for the Iris dataset split provided in:

- `iris_training.txt`
- `iris_test.txt`

No ML libraries are used. All counting and probability logic is implemented manually with standard C++ containers and loops.

## What the program does

- Reads training and test samples from text files
- Trains a nominal Naive Bayes model using frequency counts
- Classifies all test samples
- Prints:
  - experiment accuracy (percentage)
  - confusion matrix
- Applies **Laplace smoothing** when needed (zero conditional probability case)
- Prints smoothing probabilities **before** and **after** smoothing
- If no natural smoothing case appears in test predictions, demonstrates smoothing on attribute #1 (assignment fallback requirement)
- Supports repeated **manual classification** from terminal input

## Dataset format

Each line is:

`attr1 attr2 attr3 attr4 class_label`

Example:

`5,1 3,5 1,4 0,2 Iris-setosa`

Note: this implementation treats each attribute token as a **categorical value**.

## Build and run

### CLion

- Open project
- Build target: `NaiveBayesClassifier1`
- Run target: `NaiveBayesClassifier1`

### CMake (terminal)

```bash
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug --target NaiveBayesClassifier1
./cmake-build-debug/NaiveBayesClassifier1
```

(On Windows, executable name is `NaiveBayesClassifier1.exe`.)

## Manual classification mode

After test evaluation, the program enters interactive mode:

- Enter attribute values separated by spaces
- Number of required values is detected dynamically from training data
- Type `q` to quit

Example input:

`5,1 3,5 1,4 0,2`

## Project structure

- `naiveBayes.cpp` - main implementation (training, prediction, smoothing, evaluation, manual mode)
- `iris_training.txt` - training split
- `iris_test.txt` - test split
- `CMakeLists.txt` - build configuration

## Implementation notes

- Classification rule:
  - choose class with maximum score  
    `P(Y = c) * Π P(Xi = xi | Y = c)`
- Smoothing used:
  - unsmoothed: `count / classCount`
  - smoothed (Laplace): `(count + 1) / (classCount + distinctValuesForAttribute)`

## Author

Student project for Naive Bayes classification assignment.

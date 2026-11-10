# Exercise 4.1 — Feature selection, cross validation and overfitting

Shows why feature selection must be done inside the cross validation.
The data are pure noise: 1000 random features for 25 subjects (10 with
label -1, 15 with label +1), so no classifier can really do better than
chance. A linear SVM is evaluated with leave-one-out cross validation in
three ways.

## Files

| File | Purpose |
|---|---|
| `overfittingTest.m` | MATLAB main script (needs `fitcsvm`, Statistics and Machine Learning Toolbox) |
| `similarityMeasure.m` | Squared Pearson correlation between a feature and the labels |
| `overfitting_test.py` | Python port of both files |

The task description is in `../HW4_2020.pdf` (Θέμα 1).

## Run

MATLAB, from this folder:

```
overfittingTest
```

GNU Octave has no `fitcsvm`, so the MATLAB script does not run there as it is.

Python (needs `numpy`, `matplotlib` and `scikit-learn`):

```
python overfitting_test.py --seed 0
python overfitting_test.py --seed 0 --runs 20 --save-dir figures
```

`--runs` repeats the experiment with new random features; the sizes of
the experiment can be changed with `--num-features`, `--num-selected`,
`--num-positive` and `--num-negative`. A bar chart of the three accuracies
opens at the end; `--save-dir` also writes it as a PNG file.

## Pipeline

1. No feature selection: the SVM is trained on all 1000 features.
2. Selection inside the cross validation: in every fold the 100 features
   most correlated with the labels are chosen from the 24 training
   subjects only.
3. Selection outside the cross validation: the 100 features are chosen
   once, from all 25 subjects, and the cross validation follows.

## Results

Python, 20 runs (`--seed 1 --runs 20`):

| Procedure | Mean accuracy | Range |
|---|---|---|
| No feature selection | 54.4 % | 44 – 72 % |
| Selection inside the cross validation | 50.8 % | 32 – 76 % |
| Selection outside the cross validation | 100 % | 100 – 100 % |

The first two procedures stay near chance, as they should on noise. The
third reports 100 %: the left-out subject took part in choosing the
features, so the test is no longer independent of the training. With 1000
random features and only 25 subjects, some features correlate with the
labels by accident, and selecting them with all subjects carries that
accident into every fold. The reviewer was right: feature selection belongs
inside the cross validation.

## Notes

- The features are random, so single-run accuracies vary by several
  subjects (one subject is 4 %). Use `--seed` in Python for reproducible
  numbers; the MATLAB script draws new features on every run.
- The Python SVM is `sklearn.svm.SVC(kernel='linear', C=1)`, the
  counterpart of `fitcsvm` with its default settings.
- On the same feature matrix the Python script gives the same three
  accuracies as the MATLAB code run in Octave with a stand-in linear SVM
  (5 matrices checked).

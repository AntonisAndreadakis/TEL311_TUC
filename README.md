# TEL311_TUC

Coursework for **TEL311 — Statistical Modeling and Pattern Recognition**
(Technical University of Crete, 2020): four project sets, each with the
assignment, the code and a report in Greek.

The exercises were written in MATLAB. In November 2026 the code was
reviewed, corrected and tested, a Python version was added next to every
MATLAB exercise, and the reports were updated to match.

## Contents

| Set | Folder | Topics | Assignment | Report |
|---|---|---|---|---|
| 1 | `1ο Project` | PCA, Fisher/LDA, feature extraction and Bayes rule on MNIST, Bayes decision boundaries | `HW1_2020_v2.pdf` | `anafora1.pdf` |
| 2 | `2o Project` | Logistic regression, regularisation, naive Bayes, Bayesian estimation, SVM | `HW2_2020.pdf` | `2013030059_Andreadakis.pdf` |
| 3 | `3ο Project` | k-NN text classification, k-means, Gaussian mixtures with EM | not in the repo | `2013030059_Andreadakis.pdf` |
| 4 | `4o Project` | Feature selection and overfitting, logistic regression with back-propagation, CNN on Fashion-MNIST | `HW4_2020.pdf` | `2013030059_Andreadakis.pdf` |

Each set folder contains:

- one folder per exercise (`exerciseN_M`) with the MATLAB files, the Python
  version, the data and a short `README.md` (what it does, how to run it,
  key results);
- the report PDF and a `report_source` folder with the HTML source, the
  figures and the script that builds the PDF.

## Exercises

| Exercise | Subject | MATLAB main script | Python script |
|---|---|---|---|
| 1.1 | PCA on 2-D data and on faces | `ex1_1_pca.m` | `ex1_1_pca.py` |
| 1.3 | Fisher discriminant, LDA vs PCA on iris | `ex1_3_lda.m` | `ex1_3_lda.py` |
| 1.4 | Aspect ratio of MNIST digits, Bayes posteriors | `exercise1_4.m` | `exercise1_4.py` |
| 1.5 | Bayes decision boundaries for two Gaussians | `exercise1_5.m` | `exercise1_5.py` |
| 2.1 | Logistic regression | `My_logisticRegression.m` | `logistic_regression.py` |
| 2.2 | Regularised logistic regression | `ex2_2_regularizedLogisticRegression.m` | `regularized_logistic_regression.py` |
| 2.4 | Bernoulli naive Bayes on digits | `ex_2_4.m` | `ex_2_4.py` |
| 2.5 | Bayesian estimation of a Gaussian mean | `ex_2_5.m` | `ex_2_5.py` |
| 2.6 | SVM solved analytically (KKT) | `ex_2_6a.m`, `ex_2_6b.m` | `ex_2_6a.py`, `ex_2_6b.py` |
| 2.7 | SVM by quadratic programming | `svm_exercise1.m`, `svm_exercise2.m`, `svm_exercise3.m` | `svm_exercise1.py`, `svm_exercise2.py`, `svm_exercise3.py` |
| 3.1 | k-NN classification of web pages | `ex3_1.m` | `ex3_1.py` |
| 3.2 | k-means and image compression | `ex3_kmeans.m` | `ex3_kmeans.py` |
| 3.3 | Gaussian mixture model with EM | `test_gmms.m` | `test_gmms.py` |
| 4.1 | Feature selection inside vs outside cross validation | `overfittingTest.m` | `overfitting_test.py` |
| 4.2 | Logistic regression with forward/backward pass | `matlab/logisticRegression.m` | `python/logisticRegression.py` |
| 4.3 | Dense network and CNN on Fashion-MNIST | (Python only) | `fashion.py`, `fashion_cnn.py` |

Themes 1.2, 1.6, 2.3 and parts of others are theory questions; they are
answered in the reports and have no code.

## Running the code

### MATLAB

Open the exercise folder in MATLAB and run the main script, for example:

```
cd('3ο Project/exercise3_3')
test_gmms
```

Toolboxes used: Statistics and Machine Learning (`mvnpdf`, `kmeans`,
`fisheriris`, `fitcsvm`, `normrnd`) and Optimization (`fminunc`,
`quadprog`).

Most scripts also run in GNU Octave with the `statistics` package
(`pkg load statistics`). The exceptions are the scripts that need
`fitcsvm` (4.1) and `quadprog` (2.7).

### Python

Python 3.10 or newer. Install the packages once:

```
pip install -r requirements.txt
```

`tensorflow` is only needed for exercise 4.3 and `cvxopt` only for 2.7;
they can be left out if those exercises are not run.

Run a script from anywhere, for example:

```
python "3ο Project/exercise3_1/ex3_1.py" -r 300 -d 2 --seed 0
```

Every script accepts `--help`. Most accept `--save-dir DIR` to save the
figures as PNG files instead of only showing them.

### Tests

The Python versions have tests in `tests/`:

```
python -m pytest                 # everything
python -m pytest -m "not slow"   # skip the slow tests
```

The tests check the functions against known values and mathematical
properties (for example analytic against numerical gradients). Tests that
need an optional package (`tensorflow`, `cvxopt`, `scikit-learn`) are
skipped when it is not installed.

The same suite runs on GitHub Actions for every pull request and every
push to `main` (`.github/workflows/tests.yml`).

### Reports

To rebuild a report after editing its `report_source/report.html`
(needs Node.js with Playwright and Chromium):

```
cd "3ο Project/report_source"
node build.js
```

The reports as submitted in 2020 are in the git history.

## How the code was checked

- The MATLAB code was run in GNU Octave 8.4, not in MATLAB itself.
  MATLAB-only behaviour (figure appearance, `quadprog`, `fitcsvm`,
  `fminunc` stopping points) was not verified.
- The Python versions were compared number by number with the Octave runs
  of the MATLAB code. Known differences: eigenvector signs, and results
  that depend on random initialisation or on where an optimiser stops
  (2.2 with lambda = 0).
- The SVM exercises 2.7 could not use `quadprog` in Octave. The MATLAB
  logic was checked with Octave's `qp` solver, and the Python versions
  reproduce the same numbers with `cvxopt`.
- Exercise 4.3 was only run for a few epochs; the full trainings and the
  optimizer comparison were not repeated, and the report keeps the 2020
  results for that exercise.

## Notes

- In exercise 2.6 B the transformation is implemented as the assignment
  prints it, Phi(x) = x - 4*||x|| - 4, with the scalar subtracted from
  both coordinates.
- In theme 1.6 the report takes lambda_ki as the cost of classifying a
  sample of class k into class i.

# Exercise 2.4 — Naive Bayes classification of digits

Classifies 28 x 28 black-and-white images of the digits 0–9 with a naive
Bayes classifier in which every pixel of every digit is a Bernoulli
variable, with parameters estimated by maximum likelihood.

## Files

| File | Purpose |
|---|---|
| `ex_2_4.m` | MATLAB version (also runs in GNU Octave) |
| `ex_2_4.py` | Python port of the same pipeline |
| `digits.mat` | `train0`..`train9` and `test0`..`test9`: 500 images per digit, one 784-pixel row each |

## Run

MATLAB / Octave, from this folder:

```
ex_2_4
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python ex_2_4.py
python ex_2_4.py --sample-digit 7 --sample-index 10 --save-dir figures
```

`--sample-digit` and `--sample-index` (1-based, as in MATLAB) choose the
training image that is displayed (default: image 43 of digit 3). The
figures are shown together at the end, and `--save-dir` also saves them as
PNG files.

## Pipeline

1. Training: the maximum-likelihood estimate of the Bernoulli parameter of
   pixel i of digit y is the mean of that pixel over the 500 training
   images of the digit. The 10 models are displayed as images.
2. Classification: for a test image x the log-likelihood of every digit is
   `sum_i x_i ln(p_yi) + (1 - x_i) ln(1 - p_yi)` and the digit with the
   highest value wins. The priors are equal and are left out.
3. The parameters are clamped to [1e-10, 1 - 1e-10], because many pixels
   have p = 0 or p = 1 exactly and ln(0) is not defined.
4. Accuracy and confusion matrix (row: true digit, column: predicted
   digit, as a fraction of the 500 test images of the true digit).

## Results

Accuracy on the 5000 test images: **0.7982** in MATLAB / Octave and in
Python; the two confusion matrices are identical.

Fraction of every digit that is classified correctly:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Correct | 0.882 | 0.944 | 0.752 | 0.832 | 0.742 | 0.692 | 0.828 | 0.792 | 0.698 | 0.820 |

## Notes

- The most frequent confusions are 4 → 9 (0.188), 5 → 3 (0.134),
  9 → 4 (0.100) and 8 → 3 (0.098): digits with similar shapes.
- MATLAB opens one figure per digit for the models; the Python script
  shows the 10 models in one figure.

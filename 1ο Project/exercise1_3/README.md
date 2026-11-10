# Exercise 1.3 — Linear Discriminant Analysis vs PCA

Reduces 2-D samples of two classes to one dimension with Fisher's linear
discriminant and with PCA, and reduces the Fisher iris data from 4 to 2
features with multiclass LDA.

## Files

| File | Purpose |
|---|---|
| `ex1_3_lda.m` | MATLAB main script (also runs in GNU Octave) |
| `fisherLinearDiscriminant.m` | 2-class Fisher discriminant |
| `myLDA.m` | Multiclass LDA |
| `projectDataLDA.m`, `recoverDataLDA.m` | Projection on the LDA vectors and back |
| `myPCA.m`, `projectDataPCA.m`, `recoverDataPCA.m` | PCA, projection and reconstruction |
| `featureNormalize.m`, `drawLine.m` | Standardization and plotting helper |
| `ex1_3_lda.py` | Python port of the same pipeline (all helpers in one file) |
| `ex1_3_data1.mat` | 14 samples with 2 features (`X`) and their class (`c`) |

## Run

MATLAB / Octave, from this folder (pauses after every step, press enter).
Octave needs the `statistics` package for `fisheriris.mat`
(`pkg load statistics`):

```
ex1_3_lda
```

Python (needs `numpy`, `scipy` and `matplotlib`, plus `scikit-learn` only
to load the iris data):

```
python ex1_3_lda.py
python ex1_3_lda.py --save-dir figures
```

`-k` is the number of principal components kept in the 2-D example
(default 1), `--new-dim` the dimension after LDA on the iris data
(default 2) and `--save-dir` saves every figure as PNG. If a file
`fisheriris.mat` (variables `meas`, `species`) is in this folder, the
Python script reads it and does not need scikit-learn.

## Pipeline

1. Standardize the features.
2. Two classes: `v = inv(Sw) * (mu1 - mu2)` scaled to unit norm, projection
   `Z = X * v'`, reconstruction on the line of `v`.
3. PCA on the same samples: projection on the first principal component.
4. Iris: priors, class means, global mean, `Sw`, `Sb`, eigenvectors of
   `inv(Sw) * Sb` for the 2 largest eigenvalues, projection of the samples.

## Key results

Identical in Octave 8.4 and Python, up to the sign of the eigenvectors.

| Quantity | Value |
|---|---|
| Fisher direction `v` | [-0.5602 0.8284] |
| First principal component | ±[0.7071 0.7071] |
| PCA eigenvalues | 1.4650, 0.5350 |
| Iris, 1st LDA vector | ±[0.1513 0.1473 -0.8560 -0.4719] |
| Iris, 2nd LDA vector | ±[0.0069 0.3279 -0.5717 0.7521] |

## Notes

- The Fisher direction separates the two classes completely (class 1
  projects to 0.08 ... 1.49, class 2 to -1.36 ... -0.13). The first
  principal component follows the direction of largest variance, where the
  projections of the two classes overlap.
- On the iris data setosa is far from the other two classes on the first
  LDA axis; versicolor and virginica touch but stay mostly apart.
- The MATLAB script prints only progress messages; the Python script also
  prints the vectors above.
- The 2nd LDA vector has the opposite sign in Octave and in SciPy, so the
  iris plot after LDA is mirrored vertically. This is not an error.
- `myPCA.m` of this exercise uses `cov(X)` (divisor m - 1), while the one
  of exercise 1.1 divides by m. The Python port follows each one. The
  eigenvectors are the same either way.

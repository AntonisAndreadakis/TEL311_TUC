# Exercise 1.1 — Principal Component Analysis

PCA by eigendecomposition of the covariance matrix, first on 50
two-dimensional samples and then on 5000 face images of 32x32 pixels
(eigenfaces, reconstruction from the first K components).

## Files

| File | Purpose |
|---|---|
| `ex1_1_pca.m` | MATLAB main script (also runs in GNU Octave) |
| `featureNormalize.m`, `myPCA.m`, `projectData.m`, `recoverData.m` | MATLAB helpers: standardization, PCA, projection, reconstruction |
| `displayData.m`, `drawLine.m` | MATLAB plotting helpers |
| `ex1_1_pca.py` | Python port of the same pipeline (all helpers in one file) |
| `ex1_1_data1.mat` | 50 samples with 2 features |
| `ex1_1_faces.mat` | 5000 faces, one 1024-pixel row per face |

## Run

MATLAB / Octave, from this folder (pauses after every step, press enter):

```
ex1_1_pca
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python ex1_1_pca.py
python ex1_1_pca.py --face-k 100 10 50 200 --save-dir figures
```

`-k` is the number of components kept for the 2-D samples (default 1),
`--face-k` the numbers of eigenfaces used for the reconstructions and
`--save-dir` saves every figure as PNG. The Python script does not pause:
all the figures are shown together at the end.

## Pipeline

1. Standardize every feature (zero mean, unit standard deviation).
2. Covariance matrix `Sigma = (1/m) * X' * X`, eigenvectors `U` and
   eigenvalues `S` sorted by descending eigenvalue.
3. Projection `Z = X * U(:, 1:K)`, reconstruction `X_rec = Z * U(:, 1:K)'`.
4. Faces: show 100 faces, the first 36 eigenfaces and the faces recovered
   from K = 100, 10, 50, 200 eigenfaces.

## Key results

Identical in Octave 8.4 and Python, up to the sign of the eigenvectors.

| Quantity | Value |
|---|---|
| Top eigenvector `U(:,1)` | ±[0.707107 0.707107] |
| Variance share of the two components | 0.867765, 0.132235 |
| Projection of the first sample (K = 1) | ±1.481274 |
| Approximation of the first sample | [-1.047419 -1.047419] |
| Faces, largest eigenvalues | 336.96, 130.91, 54.68 |
| Faces, variance lost with K = 10 / 50 / 100 / 200 | 32.73 % / 13.21 % / 6.81 % / 2.69 % |

## Notes

- An eigenvector is defined up to its sign: Octave returns
  `U(:,1) = [-0.707 -0.707]` and NumPy `[0.707 0.707]`, so the projection
  of the first sample is 1.481274 in Octave and -1.481274 in Python. The
  reconstruction is the same in both.
- The first eigenfaces look like blurred faces (lighting, face outline);
  the later ones hold finer details. With K = 100 the faces are easy to
  recognize although 1024 numbers per face became 100.
- The Python script gives every plot its own figure. The MATLAB script
  draws the faces on figure 2, over the plot of the normalized samples.

# Exercise 3.3 — Gaussian Mixture Model with EM

Fits a 3-component Gaussian Mixture Model to the Fisher iris data with the
Expectation-Maximization algorithm and compares the resulting clustering
with k-means.

## Files

| File | Purpose |
|---|---|
| `test_gmms.m` | Main script: fits the GMM, runs k-means, prints accuracies, plots |
| `fitGMMs.m` | EM algorithm (k-means initialisation, E-step, M-step, convergence check) |
| `classify_with_gmms.m` | Assigns one example to its most probable component |
| `gaussLogPdf.m` | Log-density of a multivariate Gaussian |
| `clusterAccuracy.m` | Matches cluster numbers to classes and returns the accuracy |
| `test_gmms.py` | Python port of the script and all its helpers, with its own k-means |

## Run

From this folder, in MATLAB (needs the Statistics and Machine Learning
Toolbox) or GNU Octave (needs the `statistics` package):

```
test_gmms
```

Python (needs `numpy`, `scipy`, `matplotlib`, and `scikit-learn` only for
loading the iris data):

```
python test_gmms.py --seed 0
python test_gmms.py --seed 0 --save-dir figures
```

`--replicates` sets the number of k-means restarts (default 5),
`--max-iters` and `--tol` the EM stopping rule (defaults 200 and 1e-6).
Both figures open at the end; `--save-dir` also writes them as PNG files.

## Result

Typical output: GMM accuracy 96.67 %, k-means accuracy 89.33 %.

The GMM does better because each component has its own full covariance
matrix, so it can follow the elongated, overlapping versicolor and
virginica clusters; k-means only measures the distance to a centre.

## Notes

- Cluster numbers are arbitrary, so `clusterAccuracy.m` matches clusters to
  classes before counting correct assignments.
- Densities are computed in the log domain to avoid numerical underflow.
- Covariance matrices are stored as `K x d x d`, as documented in the
  function headers.
- The Python version does not call a library k-means: it implements
  k-means with k-means++ seeding and restarts, the defaults of MATLAB's
  `kmeans`. Both versions give the same accuracies, the same wrongly
  assigned flowers and the same GMM parameters (to 5 decimals or better).
- k-means has two almost equally good solutions on this data (88.67 % and
  89.33 %); the restarts make the better one, 89.33 %, the usual result.

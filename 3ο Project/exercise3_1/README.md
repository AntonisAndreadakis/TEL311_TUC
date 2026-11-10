# Exercise 3.1 — k-NN classification of web pages

Classifies university web pages (WebKB subset: course, faculty, project,
student) with k-NN on TF-IDF features, using 5-fold stratified cross
validation for k = 1, 3, 5, 10.

## Files

| File | Purpose |
|---|---|
| `ex3_1.m` + helper `.m` files | MATLAB version (also runs in GNU Octave) |
| `ex3_1.py` | Python port of the same pipeline |
| `make_report.py` | Re-runs all experiments and writes `knn_results_report.pdf` |
| `knn_results_report.pdf` | Results, observations and conclusion |
| `Data/` | Term-document matrices, one CSV per class |

## Run

MATLAB / Octave, from this folder (asks for the number of words R and the metric):

```
ex3_1
```

Python (needs `numpy` and `pandas`; `make_report.py` also needs `matplotlib`):

```
python ex3_1.py -r 300 -d 2 --seed 0
python make_report.py
```

`-d 1` is Euclidean distance, `-d 2` is cosine distance.

## Pipeline

1. Rank every word by its log-entropy weight and keep the R words with the
   lowest weight (frequent, broadly distributed words).
2. TF-IDF weight the reduced matrix and normalise each document to unit length.
3. Remove documents that contain none of the R words (all-zero vectors).
4. k-NN with majority vote; accuracy is the mean over the 5 folds.

## Notes

- On unit-length vectors Euclidean and cosine distance rank neighbours
  identically, so both metrics give the same accuracy once the empty
  documents are removed (about 71–77 %, see the report).
- If empty documents are kept, Euclidean distance breaks: they tie at
  distance 1.0 with every document and crowd out the real neighbours.
- The classes are unbalanced; always predicting `student` scores 39 %.
- Folds are random. Use `--seed` in Python for reproducible numbers; the
  MATLAB script draws new folds on every run, so its results vary by about
  one percentage point.

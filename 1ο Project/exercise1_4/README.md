# Exercise 1.4 — Aspect ratio of digits and Bayes rule

Computes the aspect ratio (width / height of the smallest rectangle that
encloses the digit) of the MNIST training images, builds 3-bin histograms
(Low, Medium, High) for the digits 1 (class C1) and 2 (class C2) and
applies the Bayes rule to an image with a low aspect ratio.

## Files

| File | Purpose |
|---|---|
| `exercise1_4.m` | MATLAB main script (also runs in GNU Octave) |
| `computeAspectRatio.m` | Aspect ratio of one image |
| `findBin.m` | Bin of a value, given the bin ends |
| `exercise1_4.py` | Python port of the same pipeline (helpers in the same file) |
| `mnist.mat` | MNIST: `trainX` (60000 x 784), `trainY`, `testX`, `testY` |
| `data/mnist.mat` | The same file, in the folder named by the assignment |

## Run

MATLAB / Octave, from this folder:

```
exercise1_4
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python exercise1_4.py
python exercise1_4.py --c1 1 --c2 2 --save-dir figures
```

`--c1` and `--c2` are the digits of the two classes and `--save-dir` saves
every figure as PNG. Both versions read `mnist.mat` of this folder.

## Pipeline

1. Aspect ratio of all 60000 training images; print the minimum and maximum.
2. Split the range of the aspect ratios of C1 and C2 in three equal bins
   and count the images of every class per bin.
3. Priors `P(C)`, likelihoods `P(bin|C)` and evidence `P(bin)` from the counts.
4. Posteriors `P(C|X=L) = P(L|C) * P(C) / P(L)`.

## Key results

Identical in Octave 8.4 and Python.

| Quantity | Value |
|---|---|
| Min / max aspect ratio (all digits, and C1 with C2) | 0.1000 / 2.2222 |
| Bin ends | 0.1000, 0.8074, 1.5148, 2.2222 |
| Counts of C1 (L, M, H) | 6393, 349, 0 |
| Counts of C2 (L, M, H) | 1473, 4402, 83 |
| P(C1), P(C2) | 0.5309, 0.4691 |
| P(L\|C1), P(M\|C1), P(H\|C1) | 0.9482, 0.0518, 0.0000 |
| P(L\|C2), P(M\|C2), P(H\|C2) | 0.2472, 0.7388, 0.0139 |
| P(L), P(M), P(H) | 0.6194, 0.3741, 0.0065 |
| P(C1\|X=L), P(C2\|X=L) | 0.8127, 0.1873 |

## Notes

- The digit 1 is narrow, so almost all its images fall in the Low bin; an
  image with a low aspect ratio is a 1 with probability 0.81.
- The bin counts are printed only by the Python script (the MATLAB script
  shows them in the bar charts).

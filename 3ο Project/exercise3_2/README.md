# Exercise 3.2 — K-Means clustering and image compression

Implements K-Means, runs it on a small 2-D dataset, and then uses it to
compress an image to 16 colours.

## Files

| File | Purpose |
|---|---|
| `ex3_kmeans.m` | MATLAB main script (also runs in GNU Octave) |
| `findClosestCentroids.m` | Assigns every example to its closest centroid |
| `computeCentroids.m` | Moves every centroid to the mean of its examples |
| `kMeansInitCentroids.m` | Picks K random examples as initial centroids |
| `runkMeans.m` | The K-Means loop (assignment step, update step) |
| `plotProgresskMeans.m`, `plotDataPoints.m`, `drawLine.m` | Plot helpers |
| `ex3_kmeans.py` | Python port of the script and all its helpers |
| `ex3data1.mat` | 300 two-dimensional examples |
| `peppers.png` | Image to compress (384 x 512 pixels) |
| `bird_small.png`, `bird_small.mat` | Alternative, smaller image |

## Run

MATLAB / Octave, from this folder (press enter at every pause):

```
ex3_kmeans
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python ex3_kmeans.py --seed 0
python ex3_kmeans.py --K 8 --max-iters 20 --image bird_small.png
python ex3_kmeans.py --seed 0 --save-dir figures
```

`--K` is the number of colours (default 16), `--max-iters` the number of
K-Means iterations (default 10). The Python script does not pause; all
figures open at the end, and `--save-dir` also writes them as PNG files.

## Pipeline

1. Assign the examples to the initial centroids `[3 3; 6 2; 8 5]`.
2. Move the centroids to the mean of their examples.
3. Run 10 iterations of K-Means and plot the path of every centroid.
4. Treat every pixel of the image as a point in RGB space and run K-Means
   with K = 16, starting from 16 randomly chosen pixels.
5. Replace every pixel by the colour of its closest centroid and show the
   original and the compressed image side by side.

## Results

| Quantity | Value |
|---|---|
| Closest centroids of the first 3 examples | 1, 3, 2 |
| Centroids after one update | (2.428301, 3.157924), (5.813503, 2.633656), (7.119387, 3.616684) |
| Centroids after 10 iterations | (1.953995, 5.025570), (3.043671, 1.015410), (6.033667, 3.000525) |
| Cluster sizes after 10 iterations | 98, 102, 100 |

The MATLAB and the Python version give identical numbers for parts 1–3.
Part 4 starts from random pixels, so the 16 colours change from run to run
(mean squared error per colour value of about 0.003 on `peppers.png`);
started from the same 16 pixels, both versions give the same colours.

## Notes

- The compressed image needs 4 bits per pixel (an index into 16 colours)
  instead of 24, plus the small colour table: about 6 times smaller.
- A centroid that loses all its examples is re-seeded with a random example.
- Use `--seed` in Python for reproducible colours.

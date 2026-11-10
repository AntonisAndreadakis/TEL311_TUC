"""
K-Means Clustering and image colour compression (Python port of
ex3_kmeans.m and its helpers).

Pipeline
--------
1. Find closest centroids: assign the examples of ex3data1.mat to the
   initial centroids [3 3; 6 2; 8 5] and print the first 3 assignments.
2. Compute means: move every centroid to the mean of its examples and
   print the new centroids.
3. K-Means clustering: run 10 iterations of K-Means on ex3data1.mat from
   the same initial centroids and plot the path of the centroids.
4. K-Means clustering on pixels: treat every pixel of peppers.png as a
   point in RGB space and run K-Means with K = 16 colours, starting from
   K randomly chosen pixels.
5. Image compression: replace every pixel by the colour of its closest
   centroid and show the original and the compressed image side by side.

Differences with respect to the MATLAB code
--------------------------------------------
* Centroid indices are 0-based inside the code (Python convention); they
  are printed 1-based so that the output matches the MATLAB script.
* The MATLAB script pauses after every part and after every plotted
  iteration; here nothing pauses and all figures are shown at the end.
* The random initial centroids of part 4 come from numpy's generator, so
  the 16 colours differ from a MATLAB run even with the same seed number.
  With the same initial centroids both versions give the same result.

Usage
-----
    python ex3_kmeans.py [--K 16] [--max-iters 10] [--image peppers.png]
                         [--seed 0] [--save-dir DIR]

Needs numpy, scipy and matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.io import loadmat

DATA_DIR = Path(__file__).parent


def find_closest_centroids(X, centroids):
    """Index of the closest centroid for every row of X
    (findClosestCentroids.m). Returns 0-based indices in 0..K-1; ties go to
    the centroid with the lowest index, as with MATLAB's min()."""
    dist = np.empty((X.shape[0], centroids.shape[0]))
    for i, c in enumerate(centroids):
        dist[:, i] = ((X - c) ** 2).sum(axis=1)
    return dist.argmin(axis=1)


def compute_centroids(X, idx, K, rng):
    """New centroids as the mean of the examples assigned to each one
    (computeCentroids.m). An empty cluster is re-seeded with a randomly
    chosen example, as in the MATLAB code."""
    centroids = np.zeros((K, X.shape[1]))
    for i in range(K):
        members = idx == i
        if members.any():
            centroids[i] = X[members].mean(axis=0)
        else:
            centroids[i] = X[rng.integers(X.shape[0])]
    return centroids


def kmeans_init_centroids(X, K, rng):
    """K randomly chosen, distinct examples of X as initial centroids
    (kMeansInitCentroids.m)."""
    randidx = rng.permutation(X.shape[0])
    return X[randidx[:K]].copy()


def draw_line(ax, p1, p2, **kwargs):
    """Draw a line from point p1 to point p2 (drawLine.m)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def plot_data_points(ax, X, idx, K):
    """Scatter plot of X, coloured by the assigned centroid
    (plotDataPoints.m)."""
    palette = plt.cm.hsv(np.linspace(0, 1, K + 1, endpoint=False))
    ax.scatter(X[:, 0], X[:, 1], s=15, facecolors="none",
               edgecolors=palette[idx])


def plot_progress_kmeans(ax, X, centroids, previous, idx, K, i):
    """Show the progress of K-Means on 2-D data: the examples coloured by
    cluster, the centroids as black crosses and a line from the previous
    position of every centroid (plotProgresskMeans.m). i is the 1-based
    iteration number."""
    plot_data_points(ax, X, idx, K)
    ax.plot(centroids[:, 0], centroids[:, 1], "x", color="k",
            markersize=10, markeredgewidth=3)
    for j in range(centroids.shape[0]):
        draw_line(ax, centroids[j], previous[j], color="k", linewidth=1)
    ax.set_title(f"Iteration number {i}")


def run_kmeans(X, initial_centroids, max_iters, rng, plot_progress=False):
    """Run max_iters iterations of K-Means from initial_centroids
    (runkMeans.m). Returns (centroids, idx, fig); fig is the progress
    figure, or None when plot_progress is False."""
    K = initial_centroids.shape[0]
    centroids = np.array(initial_centroids, dtype=float)
    previous_centroids = centroids
    idx = np.zeros(X.shape[0], dtype=int)

    fig = None
    if plot_progress:
        fig, ax = plt.subplots()

    for i in range(1, max_iters + 1):
        print(f"K-Means iteration {i}/{max_iters}...", flush=True)

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        if plot_progress:
            # Only the newest assignment is kept visible; the centroid
            # crosses and their paths accumulate, as with MATLAB's hold on
            for old in ax.collections:
                old.remove()
            plot_progress_kmeans(ax, X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K, rng)

    return centroids, idx, fig


def load_image(filename):
    """Read an image as a float array with values in the range 0 - 1
    (imread followed by the division by 255 in ex3_kmeans.m)."""
    A = imread(filename)
    if A.dtype == np.uint8:
        A = A / 255.0
    else:
        # matplotlib returns float32 for PNG files: go back to the exact
        # 8-bit levels so that the values equal MATLAB's double(A) / 255
        A = np.round(A.astype(float) * 255.0) / 255.0
    return A[:, :, :3]  # drop the alpha channel, if any


def main(argv=None):
    """Run the script; argv defaults to the command line (sys.argv)."""
    parser = argparse.ArgumentParser(
        description="K-Means clustering and image colour compression")
    parser.add_argument("--K", type=int, default=16,
                        help="number of colours of the compressed image")
    parser.add_argument("--max-iters", type=int, default=10,
                        help="number of K-Means iterations")
    parser.add_argument("--image", default="peppers.png",
                        help="image to compress (file name in this folder "
                             "or a path), e.g. bird_small.png")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for the initial colours of part 4")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)
    if args.K < 1 or args.max_iters < 1:
        parser.error("--K and --max-iters must be at least 1")
    rng = np.random.default_rng(args.seed)

    # ================= Part 1: Find Closest Centroids ====================
    print("Finding closest centroids.\n")

    X = loadmat(DATA_DIR / "ex3data1.mat")["X"]
    K = 3
    initial_centroids = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])

    idx = find_closest_centroids(X, initial_centroids)

    print("Closest centroids for the first 3 examples: ")
    print("".join(f" {i + 1}" for i in idx[:3]))
    print("(the closest centroids should be 1, 3, 2 respectively)")

    # ===================== Part 2: Compute Means =========================
    print("\nComputing centroids means.\n")

    centroids = compute_centroids(X, idx, K, rng)

    print("Centroids computed after initial finding of closest centroids: ")
    for c in centroids:
        print(f" {c[0]:f} {c[1]:f} ")
    print("\n(the centroids should be")
    print("   [ 2.428301 3.157924 ]")
    print("   [ 5.813503 2.633656 ]")
    print("   [ 7.119387 3.616684 ]\n")

    # =================== Part 3: K-Means Clustering ======================
    print("\nRunning K-Means clustering on example dataset.\n")

    max_iters = 10
    centroids, idx, fig_progress = run_kmeans(X, initial_centroids, max_iters,
                                              rng, plot_progress=True)
    print("\nK-Means Done.\n")

    # ============= Part 4: K-Means Clustering on Pixels ===============
    print("\nRunning K-Means clustering on pixels from an image.\n")

    image_path = Path(args.image)
    if not image_path.is_absolute() and not image_path.exists():
        image_path = DATA_DIR / image_path
    A = load_image(image_path)
    img_size = A.shape

    # Every row holds the Red, Green and Blue values of one pixel
    X = A.reshape(img_size[0] * img_size[1], 3)
    if args.K > X.shape[0]:
        parser.error(f"--K must not exceed the number of pixels ({X.shape[0]})")

    K = args.K
    initial_centroids = kmeans_init_centroids(X, K, rng)
    centroids, idx, _ = run_kmeans(X, initial_centroids, args.max_iters, rng)

    # ================= Part 5: Image Compression ======================
    print("\nApplying K-Means to compress an image.\n")

    # Find closest cluster members and map every pixel to its centroid
    idx = find_closest_centroids(X, centroids)
    X_recovered = centroids[idx].reshape(img_size[0], img_size[1], 3)

    fig_image, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(A)
    axes[0].set_title("Original")
    axes[1].imshow(X_recovered)
    axes[1].set_title(f"Compressed, with {K} colors.")
    fig_image.tight_layout()

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig_progress.savefig(args.save_dir / "kmeans_progress.png", dpi=150)
        fig_image.savefig(args.save_dir / "kmeans_compression.png", dpi=150)
        print(f"Figures saved in {args.save_dir}")

    plt.show()


if __name__ == "__main__":
    main()

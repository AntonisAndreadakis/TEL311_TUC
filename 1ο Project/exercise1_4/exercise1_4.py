"""
Aspect ratio of MNIST digits and Bayes posteriors (Python port of
exercise1_4.m and its helpers).

Pipeline
--------
1. Load the 60000 training images (28x28) and their labels from mnist.mat.
2. Compute the aspect ratio (width / height) of the smallest rectangle that
   encloses every digit and print its minimum and maximum.
3. Keep the classes C1 (digit 1) and C2 (digit 2). Split the range
   [min, max] of their aspect ratios in three equal bins (Low, Medium,
   High) and build the histograms of C1, of C2 and of both together.
4. Estimate from the counts the priors P(C1), P(C2), the likelihoods
   P(L|C), P(M|C), P(H|C) and the evidence P(L), P(M), P(H).
5. Bayes rule for a new image with a low aspect ratio:
   P(C|X=L) = P(L|C) * P(C) / P(L).

Differences with respect to the MATLAB script
---------------------------------------------
* The digits of the two classes can be chosen on the command line
  (the defaults are the digits 1 and 2 of the assignment).
* The bin counts are also printed (the MATLAB script only plots them).

Usage
-----
    python exercise1_4.py [--c1 1] [--c2 2] [--save-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_FILE = Path(__file__).parent / "mnist.mat"
BIN_NAMES = ("Low", "Medium", "High")
NUM_BINS = len(BIN_NAMES)


def compute_aspect_ratio(image):
    """Aspect ratio (width / height) of the smallest rectangle that encloses
    the digit, i.e. all the non-zero pixels (mirrors computeAspectRatio.m).

    image: 2-D array (rows x columns). Returns NaN for a blank image.
    """
    rows = np.flatnonzero(np.any(image != 0, axis=1))  # rows with ink
    cols = np.flatnonzero(np.any(image != 0, axis=0))  # columns with ink
    if rows.size == 0:  # blank image: no rectangle
        return np.nan
    # +1 because both the first and the last row/column belong to the digit
    h = rows[-1] - rows[0] + 1
    w = cols[-1] - cols[0] + 1
    return w / h


def find_bin(value, bin_ends):
    """Index (0-based) of the bin where value belongs (mirrors findBin.m,
    which returns the same index 1-based).

    Bin i covers [bin_ends[i], bin_ends[i+1]); the last bin also includes
    its upper end, so that the maximum value falls in the last bin.
    """
    num_bins = len(bin_ends) - 1
    index = 0
    for i in range(num_bins):
        if value >= bin_ends[i]:
            index = i
    return index


def plot_histogram(num, counts, title):
    """Bar chart of the counts of the three bins in figure num."""
    fig, ax = plt.subplots(num=num)
    ax.bar(np.arange(1, NUM_BINS + 1), counts)
    ax.set_xticks(np.arange(1, NUM_BINS + 1))
    ax.set_xticklabels(BIN_NAMES)
    ax.set_title(title)


def finish_figures(save_dir, prefix):
    """Save every open figure as PNG in save_dir (if given) and show them."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for num in plt.get_fignums():
            plt.figure(num).savefig(save_dir / f"{prefix}_fig{num}.png", dpi=120)
        print(f"\nFigures saved in {save_dir}")
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MNIST aspect ratio histograms and Bayes posteriors")
    parser.add_argument("--c1", type=int, default=1, choices=range(10),
                        metavar="DIGIT", help="digit of class C1 (default: 1)")
    parser.add_argument("--c2", type=int, default=2, choices=range(10),
                        metavar="DIGIT", help="digit of class C2 (default: 2)")
    parser.add_argument("--save-dir", default=None, metavar="DIR",
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)
    if args.c1 == args.c2:
        parser.error("--c1 and --c2 must be different digits")

    data = loadmat(DATA_FILE)
    # every row of trainX is an image stored row by row
    images = data["trainX"].reshape(-1, 28, 28)
    labels = data["trainY"].ravel().astype(int)

    # Aspect ratio of every image
    aspect_ratios = np.array([compute_aspect_ratio(im) for im in images])
    print(f"All digits: min aspect ratio = {np.min(aspect_ratios):.4f}, "
          f"max aspect ratio = {np.max(aspect_ratios):.4f}")

    # For the histograms use the range of the images of the classes C1, C2
    is_c1 = labels == args.c1
    is_c2 = labels == args.c2
    min_aspect_ratio = np.min(aspect_ratios[is_c1 | is_c2])
    max_aspect_ratio = np.max(aspect_ratios[is_c1 | is_c2])
    print(f"C1 and C2: min aspect ratio = {min_aspect_ratio:.4f}, "
          f"max aspect ratio = {max_aspect_ratio:.4f}")

    bin_ends = np.linspace(min_aspect_ratio, max_aspect_ratio, NUM_BINS + 1)

    # Counts of the histograms (index 0: Low, 1: Medium, 2: High)
    c1_bins = np.zeros(NUM_BINS, dtype=int)
    c2_bins = np.zeros(NUM_BINS, dtype=int)
    for ratio in aspect_ratios[is_c1]:
        c1_bins[find_bin(ratio, bin_ends)] += 1
    for ratio in aspect_ratios[is_c2]:
        c2_bins[find_bin(ratio, bin_ends)] += 1
    all_bins = c1_bins + c2_bins

    plot_histogram(1, all_bins, "Aspect ratio histogram: C1 and C2")
    plot_histogram(2, c1_bins,
                   f"Aspect ratio histogram: C1 (digit {args.c1})")
    plot_histogram(3, c2_bins,
                   f"Aspect ratio histogram: C2 (digit {args.c2})")

    n1 = c1_bins.sum()  # number of images of class C1
    n2 = c2_bins.sum()  # number of images of class C2

    # Prior probabilities
    p_c1 = n1 / (n1 + n2)
    p_c2 = n2 / (n1 + n2)

    # Likelihoods P(L|C), P(M|C), P(H|C)
    p_given_c1 = c1_bins / n1
    p_given_c2 = c2_bins / n2

    # Evidence P(L), P(M), P(H)
    p_evidence = all_bins / (n1 + n2)

    # Posterior probabilities for X = L (Bayes rule)
    p_c1_given_l = p_given_c1[0] * p_c1 / p_evidence[0]
    p_c2_given_l = p_given_c2[0] * p_c2 / p_evidence[0]

    print("\nBin counts:  C1: L = {}, M = {}, H = {}".format(*c1_bins))
    print("             C2: L = {}, M = {}, H = {}".format(*c2_bins))
    print(f"\nPriors:      P(C1) = {p_c1:.4f}, P(C2) = {p_c2:.4f}")
    print("Likelihoods: P(L|C1) = {:.4f}, P(M|C1) = {:.4f}, "
          "P(H|C1) = {:.4f}".format(*p_given_c1))
    print("             P(L|C2) = {:.4f}, P(M|C2) = {:.4f}, "
          "P(H|C2) = {:.4f}".format(*p_given_c2))
    print("Evidence:    P(L) = {:.4f}, P(M) = {:.4f}, "
          "P(H) = {:.4f}".format(*p_evidence))
    print(f"Posteriors:  P(C1|X=L) = {p_c1_given_l:.4f}, "
          f"P(C2|X=L) = {p_c2_given_l:.4f}")

    finish_figures(args.save_dir, "ex1_4")


if __name__ == "__main__":
    main()

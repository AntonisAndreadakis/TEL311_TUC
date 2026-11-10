"""
Bernoulli Naive Bayes classification of digits (Python port of ex_2_4.m).

Every digit is a 28 x 28 black-and-white image stored as a row of 784
values in {0, 1}. Every pixel i of every digit y is modelled as a Bernoulli
variable with parameter p_yi, and the pixels are assumed independent.

Pipeline
--------
1. Load `digits.mat` (train0..train9 and test0..test9, 500 x 784 each) and
   show one training sample (default: sample 43 of digit 3).
2. Training: the maximum-likelihood estimate of p_yi is the mean of pixel i
   over the training images of digit y. Show the 10 models as images.
3. Classification of the test set with equal priors:
       log p(x|y) = sum_i x_i ln(p_yi) + (1 - x_i) ln(1 - p_yi)
   and the predicted digit is the y with the highest log-likelihood. The
   parameters are clamped to [eps, 1 - eps] so that ln(0) never occurs.
4. Print the accuracy (cumulative after each digit, then total) and the
   confusion matrix (row: true digit, column: predicted digit, each row
   divided by the number of test images of that digit), and plot it.

The MATLAB script has no helper functions; its three steps are the
functions `train_bernoulli`, `log_likelihoods` and `classify` here.

Usage
-----
    python ex_2_4.py [--sample-digit 3] [--sample-index 43] [--eps 1e-10]
                     [--save-dir DIR]

All figures are shown together at the end; with `--save-dir` they are also
written as PNG files in that folder.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

DATA_FILE = Path(__file__).parent / "digits.mat"
N_DIGITS = 10
IMAGE_SHAPE = (28, 28)


def load_digits(filename=DATA_FILE):
    """Return (train, test): two lists with one (n_samples x 784) float
    array per digit 0..9."""
    mat = loadmat(filename)
    train = [mat[f"train{d}"].astype(float) for d in range(N_DIGITS)]
    test = [mat[f"test{d}"].astype(float) for d in range(N_DIGITS)]
    return train, test


def to_image(row):
    """Reshape a 784-vector to the 28 x 28 image, as reshape(row,28,28)'
    does in MATLAB (the vector is stored row by row)."""
    return np.asarray(row).reshape(IMAGE_SHAPE)


def train_bernoulli(train):
    """Maximum-likelihood Bernoulli parameters: B[y, i] = mean of pixel i
    over the training images of digit y (10 x 784)."""
    return np.vstack([t.mean(axis=0) for t in train])


def log_likelihoods(X, B, eps=1e-10):
    """log p(x|y) for every row x of X and every digit y (n x 10).

    log p(x|y) = sum_i x_i ln(B[y, i]) + (1 - x_i) ln(1 - B[y, i])
    with B clamped to [eps, 1 - eps] before taking logarithms.
    """
    B = np.clip(B, eps, 1.0 - eps)
    return X @ np.log(B).T + (1.0 - X) @ np.log(1.0 - B).T


def classify(X, B, eps=1e-10):
    """Predicted digit (0..9) for every row of X: the class with the highest
    log-likelihood (equal priors; ties go to the smaller digit)."""
    return np.argmax(log_likelihoods(X, B, eps), axis=1)


def save_figures(save_dir, names):
    """Save every open figure as PNG in save_dir, using the given names."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for num, name in zip(plt.get_fignums(), names):
        plt.figure(num).savefig(save_dir / f"{name}.png", dpi=150,
                                bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Bernoulli naive Bayes classifier for digits")
    parser.add_argument("--sample-digit", type=int, default=3,
                        choices=range(N_DIGITS), metavar="0-9",
                        help="digit of the training sample to display (default 3)")
    parser.add_argument("--sample-index", type=int, default=43,
                        help="1-based index of that sample, as in MATLAB (default 43)")
    parser.add_argument("--eps", type=float, default=1e-10,
                        help="clamp of the Bernoulli parameters (default 1e-10)")
    parser.add_argument("--save-dir", default=None,
                        help="folder in which every figure is saved as PNG")
    args = parser.parse_args()

    train, test = load_digits()
    n_samples = train[args.sample_digit].shape[0]
    if not 1 <= args.sample_index <= n_samples:
        parser.error(f"--sample-index must be between 1 and {n_samples}")

    # ------------------- One training sample -------------------
    fig, ax = plt.subplots()
    im = ax.imshow(to_image(train[args.sample_digit][args.sample_index - 1]))
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Training sample {args.sample_index} of digit {args.sample_digit}")

    # ------------------- Training (MLE) -------------------
    B = train_bernoulli(train)

    # MATLAB opens one figure per digit; here the 10 models share one figure
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8))
    for digit, ax in enumerate(axes.ravel()):
        im = ax.imshow(to_image(B[digit]), vmin=0.0, vmax=1.0)
        ax.set_title(f"digit {digit}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("MLE Bernoulli Distribution of every digit")
    fig.colorbar(im, ax=axes, shrink=0.8, label="p_yi")

    # ------------------- Classification of the test set -------------------
    confusion_matrix = np.zeros((N_DIGITS, N_DIGITS))
    correct = 0
    tested = 0
    for label in range(N_DIGITS):
        pred = classify(test[label], B, args.eps)
        confusion_matrix[label] = np.bincount(pred, minlength=N_DIGITS)
        correct += int(np.sum(pred == label))
        tested += len(pred)
        print(f"Accuracy after testing digits {label} : {correct / tested:.5g}")
    print(f"Accuracy: {correct / tested:.5g}")

    # Row i: fraction of the test images of digit i classified as each digit
    confusion_matrix /= np.array([len(t) for t in test])[:, None]

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(N_DIGITS))
    ax.set_yticks(range(N_DIGITS))
    ax.set_xlabel("Classified as digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion Matrix")

    print("Confusion matrix (row: true digit 0..9, column: classified as 0..9):")
    with np.printoptions(precision=3, suppress=True, linewidth=120,
                         floatmode="fixed"):
        print(confusion_matrix)

    if args.save_dir:
        save_figures(args.save_dir,
                     ["sample_digit", "bernoulli_models", "confusion_matrix"])
    plt.show()


if __name__ == "__main__":
    main()

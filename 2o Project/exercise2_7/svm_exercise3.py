"""
Exercise 2.7, part 3: samples that are not linearly separable
(Python port of svm_exercise3.m).

Pipeline
--------
1. Read the 100 examples of twofeature2.txt (one class surrounds the
   other, so no line separates them).
2. Form the augmented samples (x1, x2) -> (x1, x2, x1^2 + x2^2); a plane
   in this 3-D space is a circle in the original one.
3. Solve the soft margin dual QP, 0 <= lambda <= C, on the augmented
   samples (cvxopt.solvers.qp, the counterpart of MATLAB's quadprog).
4. Print the number of support vectors, w, w0 and the number of
   misclassified training examples.
5. Plot the samples, the support vectors and the decision boundary
   w * [x1, x2, x1^2 + x2^2]' + w0 = 0 in the original (x1, x2) plane.

Usage
-----
    python svm_exercise3.py [--C 1] [--save-dir DIR]

Try C = 0.01, 0.1, 1, 10, 100, 1000, 10000.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import svm_dual


def positive_float(text):
    """argparse type: a float greater than zero."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("C must be positive")
    return value


def plot_boundary(ax, w, w0):
    """Zero level of w * [x1, x2, x1^2 + x2^2]' + w0 on [-1, 1] x [-1, 1].

    The MATLAB script marks the grid points where |value| < 0.01; a contour
    at level 0 draws the same curve.
    """
    x1, x2 = np.meshgrid(np.arange(-1, 1.005, 0.01), np.arange(-1, 1.005, 0.01))
    value = w[0] * x1 + w[1] * x2 + w[2] * (x1 ** 2 + x2 ** 2) + w0
    ax.contour(x1, x2, value, levels=[0], colors="k")
    ax.plot([], [], "k-", label="Decision boundary")    # legend entry


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SVM with the extra feature ||x||^2 (100 examples)")
    parser.add_argument("--C", type=positive_float, default=1,
                        help="penalty of the slack variables (default: 1)")
    parser.add_argument("--save-dir", default=None,
                        help="also save the figure as PNG in this folder")
    args = parser.parse_args(argv)

    X, y = svm_dual.load_data("twofeature2.txt")
    Xa = svm_dual.add_norm_feature(X)   # augmented dataset
    model = svm_dual.fit(Xa, y, C=args.C)
    sv, w = model["sv"], model["w"]

    print(f"C = {args.C:g}")
    print(f"Number of support vectors: {len(sv)}")
    print("Estimated weight vector:", svm_dual.fmt(w, ".4f"))
    print(f"Bias: {model['w0']:f}")
    print(f"Misclassified training examples: {model['misclassified']}")

    fig, ax = plt.subplots()
    svm_dual.plot_samples(ax, X, y, sv)
    plot_boundary(ax, w, model["w0"])
    ax.set_box_aspect(1)
    ax.set_title(f"SVM with the extra feature ||x||^2, C = {args.C:g}")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))

    if args.save_dir:
        svm_dual.save_figures([(f"svm_exercise3_C{args.C:g}", fig)],
                              args.save_dir)
    plt.show()


if __name__ == "__main__":
    main()

"""
Exercise 2.7, part 1: hard margin SVM (Python port of svm_exercise1.m).

Pipeline
--------
1. Read twofeature1.txt and leave out the last example (an outlier), so
   that the remaining 50 examples are linearly separable.
2. Solve the dual QP without an upper bound on the multipliers
   (cvxopt.solvers.qp, the counterpart of MATLAB's quadprog).
3. Print the support vectors, their multipliers, w, w0 and the margin
   width 2/||w||.
4. Plot the samples, the support vectors, the decision line and the two
   margins.

Usage
-----
    python svm_exercise1.py [--save-dir DIR]
"""

import argparse

import matplotlib.pyplot as plt

import svm_dual


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hard margin SVM (50 examples)")
    parser.add_argument("--save-dir", default=None,
                        help="also save the figure as PNG in this folder")
    args = parser.parse_args(argv)

    X, y = svm_dual.load_data("twofeature1.txt")
    X, y = X[:-1], y[:-1]               # leave out the last example
    model = svm_dual.fit(X, y, C=None)  # hard margin: no slack variables
    sv, w = model["sv"], model["w"]

    # 1-based indices, as in MATLAB
    print("Support vector indices:", svm_dual.fmt(sv + 1, "d"))
    print("Support vector weights:", svm_dual.fmt(model["lam"][sv], ".4f"))
    print("Estimated weight vector:", svm_dual.fmt(w, ".4f"))
    print(f"Bias: {model['w0']:f}")
    print(f"Width of the margin: {model['width']:f}")
    print()

    fig, ax = plt.subplots()
    svm_dual.plot_samples(ax, X, y, sv)
    svm_dual.plot_line_and_margins(ax, X, w, model["w0"])
    ax.set_box_aspect(1)
    ax.set_title(f"Hard margin SVM ({len(y)} examples)")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))

    if args.save_dir:
        svm_dual.save_figures([("svm_exercise1", fig)], args.save_dir)
    plt.show()


if __name__ == "__main__":
    main()

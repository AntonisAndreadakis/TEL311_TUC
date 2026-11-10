"""
Exercise 2.7, part 2: soft margin SVM (Python port of svm_exercise2.m).

Pipeline
--------
1. Read all 51 examples of twofeature1.txt (the last one is an outlier of
   the positive class lying among the negative examples).
2. Solve the dual QP with slack variables, 0 <= lambda <= C
   (cvxopt.solvers.qp, the counterpart of MATLAB's quadprog).
3. Print the support vectors, their multipliers, w, w0, the margin width
   2/||w|| and the number of misclassified training examples.
4. Plot the samples, the support vectors, the decision line and the two
   margins.

Usage
-----
    python svm_exercise2.py [--C 10000] [--save-dir DIR]

Try C = 0.01, 0.1, 1, 10, 100, 1000, 10000: a small C gives a wide margin
that tolerates errors, a large C approaches the hard margin solution.
"""

import argparse

import matplotlib.pyplot as plt

import svm_dual


def positive_float(text):
    """argparse type: a float greater than zero."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("C must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Soft margin SVM (51 examples)")
    parser.add_argument("--C", type=positive_float, default=10000,
                        help="penalty of the slack variables (default: 10000)")
    parser.add_argument("--save-dir", default=None,
                        help="also save the figure as PNG in this folder")
    args = parser.parse_args(argv)

    X, y = svm_dual.load_data("twofeature1.txt")    # all the examples
    model = svm_dual.fit(X, y, C=args.C)
    sv, w = model["sv"], model["w"]

    print(f"C = {args.C:g}")
    # 1-based indices, as in MATLAB
    print("Support vector indices:", svm_dual.fmt(sv + 1, "d"))
    print("Support vector weights:", svm_dual.fmt(model["lam"][sv], ".4f"))
    print("Estimated weight vector:", svm_dual.fmt(w, ".4f"))
    print(f"Bias: {model['w0']:f}")
    print(f"Width of the margin: {model['width']:f}")
    print(f"Misclassified training examples: {model['misclassified']}")
    print()

    fig, ax = plt.subplots()
    svm_dual.plot_samples(ax, X, y, sv)
    svm_dual.plot_line_and_margins(ax, X, w, model["w0"])
    ax.set_box_aspect(1)
    ax.set_title(f"Soft margin SVM ({len(y)} examples), C = {args.C:g}")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))

    if args.save_dir:
        svm_dual.save_figures([(f"svm_exercise2_C{args.C:g}", fig)],
                              args.save_dir)
    plt.show()


if __name__ == "__main__":
    main()

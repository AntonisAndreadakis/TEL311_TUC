"""
Feature selection, cross validation and overfitting (Python port of
overfittingTest.m and similarityMeasure.m).

Pipeline
--------
1. Generate purely random features (1000 per subject) for 25 subjects,
   10 labelled -1 and 15 labelled +1. The features carry no information
   about the labels, so an honest accuracy estimate must be near chance.
2. Classify without feature selection: leave-one-out cross validation of a
   linear SVM on all the features.
3. Classify with feature selection inside the cross validation: in every
   fold the 100 features most correlated with the labels are chosen from
   the 24 training subjects only.
4. Classify with feature selection outside the cross validation: the 100
   features are chosen once, from all 25 subjects, before the cross
   validation. The left-out subject has then already influenced the
   choice, and the accuracy is far too optimistic (overfitting).

Differences with respect to the MATLAB code
--------------------------------------------
* The linear SVM is sklearn.svm.SVC(kernel='linear', C=1), the counterpart
  of MATLAB's fitcsvm with its default settings (linear kernel, box
  constraint 1, no standardisation).
* The random features come from numpy, so the accuracies of one run differ
  from one MATLAB run; they vary from run to run in MATLAB as well.
* A bar chart of the three accuracies is added (the MATLAB script only
  prints them), and --runs repeats the experiment with fresh features.

Usage
-----
    python overfitting_test.py [--num-features 1000] [--num-selected 100]
                               [--num-positive 15] [--num-negative 10]
                               [--runs 1] [--seed 0] [--save-dir DIR]

Needs numpy, matplotlib and scikit-learn (only for the linear SVM).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

PROCEDURES = (
    ("none", "Classify without feature selection:"),
    ("inside", "Classify with feature selection inside the cross validation:"),
    ("outside", "Classify with feature selection outside the cross validation:"),
)


def similarity_measure(x, y):
    """Squared Pearson correlation coefficient between the vectors x and y
    (similarityMeasure.m)."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    r = (x0 @ y0) / (np.sqrt(x0 @ x0) * np.sqrt(y0 @ y0))
    return r ** 2


def select_features(features, labels, num_selected):
    """Indices of the num_selected features with the highest similarity to
    the labels (the "Feature selection" block of overfittingTest.m)."""
    r = np.array([similarity_measure(features[:, j], labels)
                  for j in range(features.shape[1])])
    # stable sort of -r: descending, ties keep the lowest index first,
    # like MATLAB's sort(r, 'descend')
    sorted_feature_indices = np.argsort(-r, kind="stable")
    return sorted_feature_indices[:num_selected]


def classify_example(train_x, train_y, test_x):
    """Train a linear SVM and classify one example (fitcsvm + predict)."""
    svm = SVC(kernel="linear", C=1.0)
    svm.fit(train_x, train_y)
    return svm.predict(test_x.reshape(1, -1))[0]


def leave_one_out_accuracy(features, labels, num_selected, selection):
    """Leave-one-out accuracy (%) of the linear SVM (one cross-validation
    block of overfittingTest.m).

    selection: 'none'    - use all the features,
               'inside'  - select the features in every fold, from the
                           training examples only,
               'outside' - select the features once, from all examples."""
    num_examples = features.shape[0]
    if selection == "outside":
        selected = select_features(features, labels, num_selected)
    else:
        selected = np.arange(features.shape[1])

    num_correctly_classified = 0
    for i in range(num_examples):
        idx = np.delete(np.arange(num_examples), i)   # leave out example i
        if selection == "inside":
            selected = select_features(features[idx], labels[idx], num_selected)
        predicted = classify_example(features[np.ix_(idx, selected)],
                                     labels[idx], features[i, selected])
        if predicted == labels[i]:
            num_correctly_classified += 1

    # Proportion of true results among the total number of cases examined
    return 100.0 * num_correctly_classified / num_examples


def plot_accuracies(accuracies):
    """Bar chart of the accuracy of the three procedures (mean over the
    runs, every single run as a dot)."""
    names = ("no feature\nselection", "selection inside\nthe CV",
             "selection outside\nthe CV")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    positions = np.arange(len(names))
    ax.bar(positions, accuracies.mean(axis=0), width=0.6,
           color="#7aa6c2", edgecolor="k")
    for p in positions:
        ax.plot(np.full(accuracies.shape[0], p), accuracies[:, p], "k.",
                markersize=7)
    ax.axhline(50, color="r", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("leave-one-out accuracy (%)")
    n_runs = accuracies.shape[0]
    ax.set_title("Linear SVM on random features"
                 + (f" (mean of {n_runs} runs)" if n_runs > 1 else ""))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main(argv=None):
    """Run the script; argv defaults to the command line (sys.argv)."""
    parser = argparse.ArgumentParser(
        description="Feature selection inside versus outside cross validation")
    parser.add_argument("--num-features", type=int, default=1000)
    parser.add_argument("--num-selected", type=int, default=100,
                        help="number of features kept by the selection")
    parser.add_argument("--num-positive", type=int, default=15,
                        help="examples with label +1 (e.g. autism)")
    parser.add_argument("--num-negative", type=int, default=10,
                        help="examples with label -1 (e.g. typically developing)")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat the experiment with new random features")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible features")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="save the figure as PNG in this folder")
    args = parser.parse_args(argv)
    if min(args.num_positive, args.num_negative) < 2:
        parser.error("at least 2 examples of every class are needed")
    if not 1 <= args.num_selected <= args.num_features:
        parser.error("--num-selected must be between 1 and --num-features")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    rng = np.random.default_rng(args.seed)

    num_examples = args.num_positive + args.num_negative
    labels = np.ones(num_examples)
    labels[:args.num_negative] = -1

    accuracies = np.zeros((args.runs, len(PROCEDURES)))
    for run in range(args.runs):
        if args.runs > 1:
            print(f"Run {run + 1} of {args.runs}\n")
        features = rng.standard_normal((num_examples, args.num_features))
        for p, (selection, message) in enumerate(PROCEDURES):
            print(message)
            accuracies[run, p] = leave_one_out_accuracy(
                features, labels, args.num_selected, selection)
            print(f"accuracy: {accuracies[run, p]:.2f}%")
            print(" ")

    if args.runs > 1:
        print(f"Mean accuracy over {args.runs} runs:")
        for p, (_, message) in enumerate(PROCEDURES):
            print(f"{message} {accuracies[:, p].mean():.2f}% "
                  f"(min {accuracies[:, p].min():.2f}%, "
                  f"max {accuracies[:, p].max():.2f}%)")

    fig = plot_accuracies(accuracies)
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_dir / "overfitting_accuracies.png", dpi=150)
        print(f"Figure saved in {args.save_dir}")

    plt.show()


if __name__ == "__main__":
    main()

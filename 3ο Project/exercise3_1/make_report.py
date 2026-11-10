"""Generate knn_results_report.pdf: results and conclusions of the k-NN
web-page classification experiments (see ex3_1.py).

Recomputes every configuration (R in {100, 300} x {Euclidean, cosine},
k in {1, 3, 5, 10}) with seed 0, both with the empty documents removed
(what ex3_1.py does) and with them kept (to show why they are removed).
Every number quoted in the text is computed here, so the report always
matches the code.

Usage:  python make_report.py      (needs numpy, pandas, matplotlib)
"""

import datetime
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import BoxStyle, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt

import ex3_1 as m

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# categorical slots 1, 2 -> the two series drawn in the chart
SERIES = {"Empty documents removed (either metric)": "#2a78d6",
          "Euclidean, empty documents kept": "#1baf7a"}
CHART_SERIES = {
    "Empty documents removed (either metric)": lambda r: (r, "Cosine", "clean"),
    "Euclidean, empty documents kept": lambda r: (r, "Euclidean", "raw"),
}

K_VALUES = (1, 3, 5, 10)
R_VALUES = (100, 300)
SEED = 0
A4 = (8.27, 11.69)

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


# ---------------------------------------------------------------- experiments
def load_data():
    vocabulary = m.get_vocabulary(m.CLASS_FILES[1])
    freqs, labels = [], []
    for label, filename in m.CLASS_FILES.items():
        f_class = m.read_term_frequencies(filename)
        freqs.append(f_class)
        labels.append(np.full(f_class.shape[0], label))
    return vocabulary, np.vstack(freqs), np.concatenate(labels)


def cv_accuracy(new_f, labels, k, dtype, seed=SEED, n_folds=5):
    rng = np.random.default_rng(seed)
    fold_ind = m.make_fold_indices(labels, n_folds, rng)
    accs = []
    for cv in range(1, n_folds + 1):
        test = fold_ind == cv
        pred = m.knn_classify(k, new_f[~test], labels[~test], new_f[test], dtype)
        accs.append(100.0 * np.mean(pred == labels[test]))
    return float(np.mean(accs))


def run_all():
    """Returns (results, facts).

    results[(r, metric, variant)] = {k: accuracy}, variant "clean" (empty
    documents removed, as in ex3_1.py) or "raw" (empty documents kept).
    Every (configuration, k) uses the same seed, hence comparable folds.
    """
    vocabulary, f, labels = load_data()
    e = m.get_word_entropy(f)
    results, n_empty = {}, {}
    for r in R_VALUES:
        _, _, raw_f = m.get_new_vocabulary(r, vocabulary, e, f)
        clean_f, clean_y, n_empty[r] = m.drop_empty_documents(raw_f, labels)
        for dtype, metric in ((1, "Euclidean"), (2, "Cosine")):
            for variant, X, y in (("clean", clean_f, clean_y),
                                  ("raw", raw_f, labels)):
                results[(r, metric, variant)] = {
                    k: cv_accuracy(X, y, k, dtype) for k in K_VALUES}
                print(f"R={r} {metric} {variant}: " + ", ".join(
                    f"k={k}: {v:.2f}%"
                    for k, v in results[(r, metric, variant)].items()))

    # supporting numbers for the observations
    r_big = max(R_VALUES)
    _, _, raw_f = m.get_new_vocabulary(r_big, vocabulary, e, f)
    clean_f, clean_y, _ = m.drop_empty_documents(raw_f, labels)
    sim = clean_f @ clean_f.T
    np.fill_diagonal(sim, -1.0)
    no_close = 100.0 * float(np.mean(sim.max(axis=1) < 0.5))

    # run-to-run spread caused by the random fold split (best metric, 5 seeds)
    spread = max(
        float(np.std([cv_accuracy(clean_f, clean_y, k, 2, seed=s)
                      for s in range(5)])) for k in K_VALUES)

    # opposite selection direction: the R words with the HIGHEST weight
    r_small = min(R_VALUES)
    top = np.argsort(e)[::-1][:r_small]
    high_f = m.tfidf(f[:, top], b_norm=True)
    high_empty = int((np.linalg.norm(high_f, axis=1) < 1e-9).sum())
    high_acc = cv_accuracy(high_f, labels, 5, 2)

    counts = np.bincount(labels)[1:]
    facts = dict(shape=f.shape, n_empty=n_empty, no_close=no_close,
                 spread=spread, high_empty=high_empty, high_acc=high_acc,
                 r_small=r_small, r_big=r_big,
                 majority=100.0 * counts.max() / counts.sum(),
                 majority_name=m.CLASS_NAMES[int(np.argmax(counts)) + 1],
                 counts=counts)
    print(facts)
    return results, facts


# ---------------------------------------------------------------- chart marks
def rounded_bar(ax, x, height, width, color, radius_px=4):
    """Column with a 4px rounded cap and a square baseline."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 1.0))
    rs_x = radius_px / (p1[0] - p0[0])
    rs_y = radius_px / (p1[1] - p0[1])
    bar = FancyBboxPatch(
        (x - width / 2, -rs_y), width, height + rs_y,
        boxstyle=BoxStyle.Round(pad=0, rounding_size=rs_x),
        mutation_aspect=rs_y / rs_x,
        facecolor=color, edgecolor="none", zorder=3)
    clip = Rectangle((x - width / 2, 0), width, height,
                     transform=ax.transData, visible=False)
    ax.add_patch(bar)
    bar.set_clip_path(clip)
    return bar


def style_panel(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    for y in range(20, 101, 20):
        ax.axhline(y, color=GRID, lw=0.8, zorder=1)
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"k = {k}" for k in K_VALUES])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)


def draw_chart(fig, results):
    shown = {name: {r: results[key(r)] for r in R_VALUES}
             for name, key in CHART_SERIES.items()}
    best = max(((name, r, k, acc) for name, by_r in shown.items()
                for r, accs in by_r.items() for k, acc in accs.items()),
               key=lambda t: t[3])
    axes = [fig.add_axes([0.10, 0.06, 0.385, 0.27]),
            fig.add_axes([0.575, 0.06, 0.385, 0.27])]
    width, offset = 0.30, 0.18
    for ax, r in zip(axes, R_VALUES):
        style_panel(ax, f"R = {r} words")
        for sign, name in zip((-1, 1), CHART_SERIES):
            for i, k in enumerate(K_VALUES):
                acc = shown[name][r][k]
                x = i + sign * offset
                rounded_bar(ax, x, acc, width, SERIES[name])
                if (name, r, k) == best[:3]:  # direct-label the best bar only
                    ax.text(x, acc + 3, f"{acc:.1f}%", ha="center",
                            fontsize=8, color=INK,
                            fontweight="bold", zorder=4)
    axes[0].set_ylabel("Total accuracy (%)", fontsize=8, color=MUTED)

    # legend (two series -> always present)
    lx = 0.10
    for name, color in SERIES.items():
        fig.patches.append(Rectangle((lx, 0.3675), 0.014, 0.0085,
                                     transform=fig.transFigure,
                                     facecolor=color, edgecolor="none"))
        fig.text(lx + 0.020, 0.367, name, fontsize=8.5, color=INK2)
        lx += 0.36


# ---------------------------------------------------------------- text helpers
def para(fig, y, text, size=9, color=INK, width=104, x=0.09, leading=0.0148):
    wrapped = textwrap.fill(text, width=width)
    fig.text(x, y, wrapped, fontsize=size, color=color, va="top", linespacing=1.45)
    return y - leading * (wrapped.count("\n") + 1) - 0.012


def heading(fig, y, text, size=11.5):
    fig.text(0.09, y, text, fontsize=size, color=INK, fontweight="bold", va="top")
    return y - 0.028


# ---------------------------------------------------------------- pages
def page_one(pdf, results, facts):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor(SURFACE)
    shape = facts["shape"]

    fig.text(0.09, 0.955, "K-NN Text Classification of Web Pages",
             fontsize=19, color=INK, fontweight="bold", va="top")
    fig.text(0.09, 0.928, "Results report - Statistical Modeling & Pattern "
             f"Recognition, Project 3 (exercise 3.1) - "
             f"{datetime.date.today():%B %d, %Y}",
             fontsize=9, color=INK2, va="top")
    fig.lines.append(plt.Line2D([0.09, 0.91], [0.914, 0.914],
                                transform=fig.transFigure, color=GRID, lw=1))

    y = heading(fig, 0.893, "Method")
    y = para(fig, y,
             f"Four classes of university web pages (course, faculty, project, "
             f"student; {shape[0]:,} documents, {shape[1]:,}-word vocabulary) are "
             f"represented by term-document matrices. Words are ranked by their "
             f"log-entropy weight and the R most broadly distributed (lowest-"
             f"weight) words are kept; the reduced matrix is TF-IDF weighted "
             f"((1 + log tf) x log(N/df)) and each document vector is normalized "
             f"to unit length. Documents that contain none of the R words "
             f"(all-zero vectors) are removed. A k-NN classifier is evaluated "
             f"with 5-fold stratified cross-validation (random seed {SEED}) for "
             f"k = 1, 3, 5, 10, with Euclidean or cosine distance, for R = 100 "
             f"and R = 300. For comparison, the same runs are repeated with the "
             f"empty documents kept.")

    y = heading(fig, y - 0.008, "Results")
    cols = [0.09, 0.19, 0.30, 0.45, 0.62, 0.77]
    fig.text((cols[2] + cols[4]) / 2 - 0.01, y, "Empty documents removed",
             fontsize=9, color=INK2, ha="center", fontweight="bold")
    fig.text((cols[4] + 0.91) / 2, y, "Empty documents kept",
             fontsize=9, color=INK2, ha="center", fontweight="bold")
    y -= 0.020
    headers = ["R words", "k neighbors", "Euclidean", "Cosine",
               "Euclidean", "Cosine"]
    for cx, h in zip(cols, headers):
        fig.text(cx, y, h, fontsize=9, color=INK2, ha="left")
    y -= 0.008
    fig.lines.append(plt.Line2D([0.09, 0.91], [y, y],
                                transform=fig.transFigure, color=BASELINE, lw=1))
    y -= 0.017
    best_val = max(acc for (_, _, var), accs in results.items()
                   if var == "clean" for acc in accs.values())
    for r in R_VALUES:
        for k in K_VALUES:
            vals = [(results[(r, met, var)][k], var)
                    for var in ("clean", "raw")
                    for met in ("Euclidean", "Cosine")]
            fig.text(cols[0], y, f"{r}" if k == K_VALUES[0] else "",
                     fontsize=9, color=INK)
            fig.text(cols[1], y, f"{k}", fontsize=9, color=INK)
            for cx, (v, var) in zip(cols[2:], vals):
                bold = ("bold" if var == "clean" and abs(v - best_val) < 1e-9
                        else "normal")
                fig.text(cx, y, f"{v:.2f}%", fontsize=9, color=INK,
                         fontweight=bold)
            y -= 0.0185
        y -= 0.004
    fig.lines.append(plt.Line2D([0.09, 0.91], [y + 0.008, y + 0.008],
                                transform=fig.transFigure, color=GRID, lw=1))
    para(fig, y - 0.004, "Total accuracy = mean accuracy over the 5 cross-"
         "validation folds. Best configuration in bold. Empty documents: "
         + ", ".join(f"{facts['n_empty'][r]} at R = {r}" for r in R_VALUES)
         + f" (of {shape[0]:,}).", size=7.5, color=MUTED)

    heading(fig, 0.415, "Accuracy by number of neighbors")
    draw_chart(fig, results)

    pdf.savefig(fig)
    plt.close(fig)


def page_two(pdf, results, facts):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor(SURFACE)

    r_small, r_big = facts["r_small"], facts["r_big"]
    clean = {(r, met): results[(r, met, "clean")]
             for r in R_VALUES for met in ("Euclidean", "Cosine")}
    raw = {(r, met): results[(r, met, "raw")]
           for r in R_VALUES for met in ("Euclidean", "Cosine")}
    max_gap = max(abs(clean[(r, "Euclidean")][k] - clean[(r, "Cosine")][k])
                  for r in R_VALUES for k in K_VALUES)
    same = ("give exactly the same accuracy in every configuration"
            if max_gap < 0.005 else
            f"differ by at most {max_gap:.2f} points in any configuration")
    c_big, c_small = clean[(r_big, "Cosine")], clean[(r_small, "Cosine")]
    e_raw = raw[(r_big, "Euclidean")]
    cos_shift = max(abs(raw[(r, "Cosine")][k] - clean[(r, "Cosine")][k])
                    for r in R_VALUES for k in K_VALUES)
    best_k = max(c_big, key=c_big.get)
    best = c_big[best_k]
    n_docs = facts["shape"][0]

    y = heading(fig, 0.955, "Observations", size=14)
    y -= 0.004
    obs = [
        ("On clean data the two metrics are the same classifier.  Every "
         "document vector has unit length, so squared Euclidean distance "
         "equals 2 - 2 x cosine similarity: both metrics rank the neighbors "
         f"in the same order. With the empty documents removed they {same}."),
        ("Empty documents break Euclidean distance.  "
         + " and ".join(f"{facts['n_empty'][r]} documents at R = {r}"
                        for r in R_VALUES)
         + " contain none of the selected words, so their vectors are all "
         "zeros. A zero vector lies at Euclidean distance exactly 1.0 from "
         "every unit vector, which is closer than any genuine neighbor whose "
         "cosine similarity is below 0.5 (distance sqrt(2 - 2 x similarity) "
         f"> 1). At R = {r_big}, {facts['no_close']:.0f}% of the documents have "
         "no neighbor above 0.5 similarity, so their 'nearest neighbors' are "
         "the empty documents, whose labels span all four classes. Keeping "
         f"them drops Euclidean accuracy at R = {r_big} to "
         f"{min(e_raw.values()):.0f}-{max(e_raw.values()):.0f}%. Because all "
         "the empty documents tie at the same distance, that figure also "
         "depends on how the sort breaks ties and is not reproducible across "
         "platforms. Cosine distance ranks a zero vector as maximally far and "
         f"moves by at most {cos_shift:.1f} points."),
        (f"More neighbors and more words both help, modestly.  At R = {r_big} "
         f"accuracy goes from {c_big[1]:.1f}% at k = 1 to {c_big[10]:.1f}% at "
         f"k = 10 (R = {r_small}: {c_small[1]:.1f}% to {c_small[10]:.1f}%). "
         "The increase is not strictly monotonic in k, and re-drawing the "
         "random folds moves a result by a standard deviation of up to "
         f"{facts['spread']:.1f} points (5 seeds), so differences of about one "
         "point between neighboring settings are not significant."),
        ("Word selection direction matters more than anything else.  Selecting "
         f"the {r_small} words with the highest entropy weight (rare, "
         f"concentrated words) leaves {facts['high_empty']:,} of {n_docs:,} "
         f"documents with empty vectors and yields {facts['high_acc']:.1f}% "
         "accuracy (cosine, k = 5, empty documents kept), below even the "
         "majority-class baseline. Selecting the lowest-weight (frequent, "
         "broadly distributed) words is what makes the results above "
         "possible."),
    ]
    for i, text in enumerate(obs, 1):
        y = para(fig, y, f"{i}.  {text}", width=112) - 0.006

    y = heading(fig, y - 0.012, "Conclusion", size=14)
    y -= 0.004
    y = para(
        fig, y,
        f"For unit-normalized TF-IDF vectors the choice between Euclidean and "
        f"cosine distance does not matter once documents with no selected "
        f"words are removed; if they are kept, cosine is the robust choice "
        f"and Euclidean fails. The best configuration found is R = {r_big} "
        f"words with k = {best_k} neighbors, reaching {best:.1f}% cross-"
        f"validated accuracy on the four-class problem. The classes are "
        f"unbalanced, so the fair baseline is always predicting the largest "
        f"class ('{facts['majority_name']}'), which scores "
        f"{facts['majority']:.1f}%.", width=112)

    fig.text(0.09, 0.045, "Generated by make_report.py from ex3_1.py "
             f"(5-fold stratified CV, seed {SEED}).", fontsize=7.5, color=MUTED)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    results, facts = run_all()
    with PdfPages("knn_results_report.pdf") as pdf:
        page_one(pdf, results, facts)
        page_two(pdf, results, facts)
    print("Wrote knn_results_report.pdf")


if __name__ == "__main__":
    main()

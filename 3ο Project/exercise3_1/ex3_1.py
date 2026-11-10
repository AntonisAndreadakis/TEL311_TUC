"""
K-NN Classification of Web-Pages (Python port of ex3_1.m and its helpers).

Pipeline
--------
1. Read the vocabulary and the term-document matrices of the 4 classes
   (course, faculty, project, student) from the semicolon-delimited CSVs.
2. Compute the entropy weight of every word and the TF-IDF features.
3. Keep the top-R most informative words (highest entropy weight).
4. Run 5-fold stratified cross-validated k-NN for k = 1, 3, 5, 10,
   with Euclidean distance or cosine distance.

Corrections with respect to the original MATLAB code
-----------------------------------------------------
* getWordEntropy: probabilities are term counts divided by the total count
  of the term over all documents (the standard log-entropy weight), not by
  the number of documents; 0*log(0) is treated as 0 instead of patching
  p with 1e-7.
* GetNewVocabulary: keeps the original's ascending sort (lowest entropy
  weight e first, i.e. the frequent, broadly-distributed words). Words with
  e close to 1 occur in only a handful of documents and yield all-zero
  feature vectors for most pages; empirically, lowest-e selection reaches
  ~73% CV accuracy at R=100 versus ~24% (chance) for highest-e.
* CosineSimilarity_Distance: uses per-row norms and returns a true
  distance (1 - similarity), so that "smallest = nearest" holds for both
  metrics inside KNN_classify.
* Empty documents: a page that contains none of the R selected words has
  an all-zero feature vector. Such a vector lies at Euclidean distance
  exactly 1.0 from every unit-length vector, so it ties with (and often
  beats) the genuine neighbours and makes the Euclidean results depend on
  how the sort breaks ties. These documents carry no information, so they
  are removed before cross validation (the count is printed), and the
  neighbour search uses a stable sort so that results are reproducible.
* Cross validation: the test/train labels are selected with the fold
  indices (the original compared labels==cv), and the fold accuracy is the
  element-wise mean(pred == truth) instead of intersect() of label sets.

Usage
-----
    python ex3_1.py -r 300 -d 2 [--seed 0]

If -r or -d is omitted, the script asks for it interactively (as the
original ex3_1.m did) and re-prompts until the value is valid. R must be
between 1 and the vocabulary size; values given on the command line are
validated against the same bounds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "Data"
CLASS_FILES = {  # class label -> file
    1: DATA_DIR / "course_TDM.csv",
    2: DATA_DIR / "faculty_TDM.csv",
    3: DATA_DIR / "project_TDM.csv",
    4: DATA_DIR / "student_TDM.csv",
}
CLASS_NAMES = {1: "course", 2: "faculty", 3: "project", 4: "student"}


def get_vocabulary(vocfilename):
    """Return the vocabulary (list of words) from the header of a TDM file."""
    header = pd.read_csv(vocfilename, sep=";", nrows=0)
    # First column is the document id, the rest are the words
    return np.array(header.columns[1:])


def read_term_frequencies(filename):
    """Read a term-document matrix, dropping the document-id column."""
    df = pd.read_csv(filename, sep=";", header=0)
    return df.iloc[:, 1:].to_numpy(dtype=float)


def get_word_entropy(f):
    """Log-entropy weight of every word from the term-frequency matrix f.

    e_j = 1 + sum_i p_ij * log2(p_ij) / log2(nDocs),  p_ij = f_ij / gf_j

    where gf_j is the total count of word j over all documents.
    e = 1: word concentrated in a single document (most informative)
    e = 0: word spread uniformly over all documents (least informative)
    """
    n_docs = f.shape[0]
    gf = f.sum(axis=0)
    gf[gf == 0] = 1  # avoid division by zero for words that never occur
    p = f / gf
    plogp = p * np.log2(p, out=np.zeros_like(p), where=p > 0)
    return 1.0 + plogp.sum(axis=0) / np.log2(n_docs)


def normalize_fea(fea, row=True):
    """Normalize each row (row=True) or column (row=False) to unit norm."""
    axis = 1 if row else 0
    norms = np.maximum(np.linalg.norm(fea, axis=axis, keepdims=True), 1e-7)
    return fea / norms


def tfidf(fea, b_norm=True):
    """TF-IDF weighting: (1 + log(tf)) * log(N / df), tf > 0.

    If b_norm, every document vector is normalized to unit norm.
    """
    n_docs = fea.shape[0]
    df = (fea > 0).sum(axis=0)
    df = np.maximum(df, 1)
    idf = np.log(n_docs / df)
    log_tf = np.log(fea, out=np.zeros_like(fea), where=fea > 0)
    tf = np.where(fea > 0, 1.0 + log_tf, 0.0)
    out = tf * idf
    return normalize_fea(out, row=True) if b_norm else out


def get_new_vocabulary(r, vocabulary, e, f):
    """Keep the r words with the lowest entropy weight (the frequent,
    broadly-distributed words — see the module docstring for why).

    Returns (new_vocabulary, new_entropy, new_f) where new_f is the
    TF-IDF-weighted term-document matrix restricted to those words.
    """
    top = np.argsort(e)[:r]
    return vocabulary[top], e[top], tfidf(f[:, top], b_norm=True)


def drop_empty_documents(new_f, labels):
    """Remove documents whose feature vector is all zeros (they contain
    none of the selected words). Returns (new_f, labels, n_removed)."""
    keep = np.linalg.norm(new_f, axis=1) > 1e-9
    return new_f[keep], labels[keep], int((~keep).sum())


def euclidean_distance(test_x, train_x):
    """Euclidean distance between one test vector and all training rows."""
    return np.sqrt(((train_x - test_x) ** 2).sum(axis=1))


def cosine_similarity_distance(test_x, train_x):
    """Cosine distance (1 - similarity) between one test vector and all
    training rows, so smaller means nearer, like the Euclidean metric."""
    num = train_x @ test_x
    denom = np.maximum(np.linalg.norm(train_x, axis=1) * np.linalg.norm(test_x), 1e-14)
    return 1.0 - num / denom


def knn_classify(k, train_f, train_y, test_f, dtype):
    """Predict a label for every row of test_f by majority vote among the
    k nearest training samples (dtype=1: Euclidean, dtype=2: cosine)."""
    dist_fn = euclidean_distance if dtype == 1 else cosine_similarity_distance
    test_z = np.empty(test_f.shape[0], dtype=train_y.dtype)
    for i, x in enumerate(test_f):
        # stable sort: ties are broken by training order, on every platform
        nearest = np.argsort(dist_fn(x, train_f), kind="stable")[:k]
        votes = train_y[nearest]
        vals, counts = np.unique(votes, return_counts=True)
        test_z[i] = vals[np.argmax(counts)]
    return test_z


def make_fold_indices(labels, n_folds, rng):
    """Stratified fold assignment: every class is split evenly over the
    folds; the remainder samples go to one random fold (as in ex3_1.m)."""
    fold_ind = np.ones(len(labels), dtype=int)
    for g in np.unique(labels):
        g_ind = np.flatnonzero(labels == g)
        rand_ind = rng.permutation(g_ind)
        n_per_fold = len(g_ind) // n_folds
        for cv in range(2, n_folds + 1):
            fold_ind[rand_ind[(cv - 1) * n_per_fold : cv * n_per_fold]] = cv
        n_rem = len(g_ind) % n_folds
        if n_rem:
            fold_ind[rand_ind[-n_rem:]] = rng.integers(1, n_folds + 1)
    return fold_ind


def ask_int(prompt, is_valid, error_msg):
    """Prompt until the user types an integer accepted by is_valid."""
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("\tPlease enter a whole number.")
            continue
        if is_valid(value):
            return value
        print(error_msg)


def main():
    parser = argparse.ArgumentParser(description="k-NN classification of web pages")
    parser.add_argument("-r", "--words", type=int, default=None,
                        help="number of words (lowest-entropy terms) to use; "
                             "asked interactively if omitted")
    parser.add_argument("-d", "--dtype", type=int, default=None, choices=(1, 2),
                        help="1: Euclidean distance, 2: cosine similarity; "
                             "asked interactively if omitted")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible folds")
    args = parser.parse_args()

    # The vocabulary size bounds R, so read the vocabulary before validating
    vocabulary = get_vocabulary(CLASS_FILES[1])
    max_words = len(vocabulary)

    if args.words is None:
        args.words = ask_int(
            f"Enter the number of words to be used (1-{max_words}): ",
            lambda r: 1 <= r <= max_words,
            f"\tThe number of words must be between 1 and {max_words}.")
    elif not 1 <= args.words <= max_words:
        parser.error(f"-r/--words must be between 1 and {max_words}, "
                     f"got {args.words}")

    if args.dtype is None:
        args.dtype = ask_int(
            "Enter (1) for Euclidean Distance and (2) for Cosine Similarity: ",
            lambda d: d in (1, 2),
            "\tPlease enter 1 or 2.")

    print(f"\tNumber of words selected: {args.words}")
    print("\tDistance Metric:",
          "Euclidean Distance" if args.dtype == 1 else "Cosine Similarity")

    # ------------------- Read and preprocess the data -------------------
    freqs, labels = [], []
    for label, filename in CLASS_FILES.items():
        f_class = read_term_frequencies(filename)
        freqs.append(f_class)
        labels.append(np.full(f_class.shape[0], label))
    f = np.vstack(freqs)
    labels = np.concatenate(labels)

    e = get_word_entropy(f)
    new_vocabulary, new_e, new_f = get_new_vocabulary(args.words, vocabulary, e, f)
    new_f, labels, n_empty = drop_empty_documents(new_f, labels)
    print(f"\tDocuments without any selected word (removed): {n_empty} "
          f"of {n_empty + len(labels)}")

    # ------------------- k-NN classification with 5-fold CV -------------
    rng = np.random.default_rng(args.seed)
    n_folds = 5
    for k in (1, 3, 5, 10):
        print(f"\tNumber of K nearest neighbors: {k}")
        fold_ind = make_fold_indices(labels, n_folds, rng)
        fold_acc = np.zeros(n_folds)
        for cv in range(1, n_folds + 1):
            test_mask = fold_ind == cv
            test_z = knn_classify(k, new_f[~test_mask], labels[~test_mask],
                                  new_f[test_mask], args.dtype)
            fold_acc[cv - 1] = 100.0 * np.mean(test_z == labels[test_mask])
            print(f"\tFold: {cv}, Accuracy: {fold_acc[cv - 1]:.2f}%")
        print(f"\tK={k} -- Total Accuracy: {fold_acc.mean():.2f}%\n")


if __name__ == "__main__":
    main()

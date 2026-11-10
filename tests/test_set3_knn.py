"""Tests of the k-NN text classification of set 3 (exercise3_1/ex3_1.py)."""

import re
import sys

import numpy as np
import pytest

from conftest import SET3, load_exercise

SCRIPT = SET3 / "exercise3_1" / "ex3_1.py"


@pytest.fixture(scope="module")
def knn():
    return load_exercise(SCRIPT)


# ----------------------------------------------------------------- entropy

def test_entropy_is_one_for_word_in_single_document(knn):
    """A word that occurs in only one document has entropy weight 1."""
    f = np.array([[3.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert knn.get_word_entropy(f)[0] == pytest.approx(1.0)


def test_entropy_is_zero_for_uniform_word(knn):
    """A word spread evenly over all documents has entropy weight 0."""
    f = np.array([[3.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    assert knn.get_word_entropy(f)[1] == pytest.approx(0.0, abs=1e-12)


def test_entropy_of_word_in_two_of_four_documents(knn):
    """A word split evenly over 2 of 4 documents has weight 1 - log2(2)/log2(4)."""
    f = np.array([[5.0], [5.0], [0.0], [0.0]])
    assert knn.get_word_entropy(f)[0] == pytest.approx(0.5)


def test_entropy_of_unused_word_is_finite(knn):
    """A word that never occurs does not produce NaN (weight 1)."""
    f = np.array([[1.0, 0.0], [2.0, 0.0]])
    e = knn.get_word_entropy(f)
    assert np.all(np.isfinite(e))
    assert e[1] == pytest.approx(1.0)


# ------------------------------------------------------------------- tfidf

def test_tfidf_rows_have_unit_norm(knn):
    """Every non-empty document has unit norm after tfidf."""
    fea = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [4.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    out = knn.tfidf(fea)
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(4))


def test_tfidf_zero_rows_stay_zero(knn):
    """A document without any word keeps an all-zero feature vector."""
    fea = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 3.0]])
    out = knn.tfidf(fea)
    assert np.all(out[1] == 0.0)
    assert np.all(np.isfinite(out))


def test_tfidf_values_without_normalization(knn):
    """Without normalization the weight is (1 + log tf) * log(N / df)."""
    fea = np.array([[2.0, 1.0], [0.0, 1.0]])
    out = knn.tfidf(fea, b_norm=False)
    expected = np.array([[(1 + np.log(2.0)) * np.log(2.0), 0.0], [0.0, 0.0]])
    assert out == pytest.approx(expected)


# ------------------------------------------------- vocabulary / empty docs

def test_new_vocabulary_keeps_lowest_weight_words(knn):
    """The r words with the smallest entropy weight are selected, in order."""
    vocabulary = np.array(["a", "b", "c", "d"])
    e = np.array([0.9, 0.1, 0.5, 0.3])
    f = np.array([[1.0, 1.0, 0.0, 2.0], [0.0, 2.0, 1.0, 0.0], [3.0, 0.0, 1.0, 1.0]])
    new_vocabulary, new_e, new_f = knn.get_new_vocabulary(2, vocabulary, e, f)
    assert list(new_vocabulary) == ["b", "d"]
    assert new_e == pytest.approx([0.1, 0.3])
    assert new_f == pytest.approx(knn.tfidf(f[:, [1, 3]]))


def test_drop_empty_documents(knn):
    """All-zero documents are removed together with their labels."""
    new_f = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    labels = np.array([1, 2, 3, 4])
    kept_f, kept_labels, n_removed = knn.drop_empty_documents(new_f, labels)
    assert n_removed == 2
    assert list(kept_labels) == [1, 3]
    assert kept_f == pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0]]))


# --------------------------------------------------------------- distances

def test_euclidean_distance_hand_example(knn):
    """Euclidean distance of the 3-4-5 triangle and of identical points."""
    train = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    d = knn.euclidean_distance(np.array([0.0, 0.0]), train)
    assert d == pytest.approx([5.0, 0.0, 1.0])


def test_cosine_distance_hand_example(knn):
    """Cosine distance is 0 for parallel, 1 for orthogonal, 2 for opposite."""
    train = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [1.0, 1.0]])
    d = knn.cosine_similarity_distance(np.array([1.0, 0.0]), train)
    assert d == pytest.approx([0.0, 1.0, 2.0, 1.0 - 1.0 / np.sqrt(2.0)])


def test_squared_euclidean_is_twice_cosine_on_unit_vectors(knn):
    """On unit vectors d_euclid^2 = 2 * d_cos, so both metrics rank alike."""
    rng = np.random.default_rng(0)
    train = knn.normalize_fea(rng.random((20, 6)))
    x = knn.normalize_fea(rng.random((1, 6)))[0]
    d_euclid = knn.euclidean_distance(x, train)
    d_cos = knn.cosine_similarity_distance(x, train)
    assert d_euclid ** 2 == pytest.approx(2.0 * d_cos)


# --------------------------------------------------------------------- knn

@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("k", [1, 3])
def test_knn_classify_separable_set(knn, k, dtype):
    """Two well separated groups are classified without error."""
    train_f = np.array([[1.0, 0.05], [1.0, 0.1], [0.9, 0.0],
                        [0.05, 1.0], [0.1, 1.0], [0.0, 0.9]])
    train_y = np.array([1, 1, 1, 2, 2, 2])
    test_f = np.array([[0.95, 0.02], [0.03, 0.8], [0.7, 0.1]])
    pred = knn.knn_classify(k, train_f, train_y, test_f, dtype)
    assert list(pred) == [1, 2, 1]


def test_knn_majority_vote_beats_single_nearest(knn):
    """With k = 3 the majority label wins over the single nearest neighbour."""
    train_f = np.array([[0.0], [1.0], [1.2], [5.0]])
    train_y = np.array([7, 9, 9, 7])
    test_f = np.array([[0.4]])
    assert list(knn.knn_classify(1, train_f, train_y, test_f, 1)) == [7]
    assert list(knn.knn_classify(3, train_f, train_y, test_f, 1)) == [9]


# ------------------------------------------------------------------- folds

def test_fold_indices_every_fold_has_every_class(knn):
    """Each of the folds contains samples of every class."""
    labels = np.repeat([1, 2, 3, 4], [10, 12, 23, 15])
    fold_ind = knn.make_fold_indices(labels, 5, np.random.default_rng(0))
    assert set(fold_ind) == {1, 2, 3, 4, 5}
    for cv in range(1, 6):
        assert set(labels[fold_ind == cv]) == {1, 2, 3, 4}


def test_fold_indices_sizes(knn):
    """Per class every fold gets n // folds samples; the remainder goes to one fold."""
    sizes = {1: 10, 2: 12, 3: 23, 4: 15}
    labels = np.repeat(list(sizes), list(sizes.values()))
    fold_ind = knn.make_fold_indices(labels, 5, np.random.default_rng(1))
    assert len(fold_ind) == len(labels)
    for g, n in sizes.items():
        counts = np.bincount(fold_ind[labels == g], minlength=6)[1:]
        assert counts.sum() == n
        assert sorted(counts) == [n // 5] * 4 + [n // 5 + n % 5]


def test_fold_indices_are_reproducible(knn):
    """The same seed gives the same folds."""
    labels = np.repeat([1, 2], [11, 14])
    a = knn.make_fold_indices(labels, 5, np.random.default_rng(3))
    b = knn.make_fold_indices(labels, 5, np.random.default_rng(3))
    assert np.array_equal(a, b)


# --------------------------------------------------------------- real data

@pytest.mark.slow
def test_real_data_r100_cosine_seed0(knn, monkeypatch, capsys):
    """R=100, cosine, seed 0: 25 documents removed and K=1 accuracy 70.83%."""
    monkeypatch.setattr(sys, "argv", ["ex3_1.py", "-r", "100", "-d", "2", "--seed", "0"])
    knn.main()
    out = capsys.readouterr().out
    removed = re.search(r"\(removed\): (\d+) of (\d+)", out)
    assert int(removed.group(1)) == 25
    assert int(removed.group(2)) == 1396
    k1 = re.search(r"K=1 -- Total Accuracy: ([0-9.]+)%", out)
    assert float(k1.group(1)) == pytest.approx(70.83, abs=0.01)

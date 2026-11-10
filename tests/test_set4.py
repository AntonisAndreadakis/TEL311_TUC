"""Tests of the Python scripts of set 4: the logistic-regression network of
exercise 4.2 and the Fashion-MNIST models of exercise 4.3."""

import re

import numpy as np
import pytest

from conftest import SET4, load_exercise

LOGREG = SET4 / "exercise4_2" / "python" / "logisticRegression.py"
FASHION_DIR = SET4 / "exercise4_3"


# ===================================================================== 4.2

@pytest.fixture(scope="module")
def logreg():
    return load_exercise(LOGREG)


def make_network(logreg, W, b, learning_rate=0.1):
    """Network with given weights instead of the random initial ones."""
    nn = logreg.NeuralNetwork(2, 1, learning_rate)
    nn.W = np.array(W, dtype=float).reshape(2, 1)
    nn.b = np.array([b], dtype=float)
    return nn


def test_sigmoid_values(logreg):
    """sigmoid(0) = 0.5, sigmoid(ln 3) = 0.75 and sigmoid(-z) = 1 - sigmoid(z)."""
    z = np.array([0.0, np.log(3.0), -np.log(3.0)])
    assert logreg.sigmoid(z) == pytest.approx([0.5, 0.75, 0.25])


def test_initial_weights_shape(logreg):
    """The network starts with W of shape (2, 1) and a zero bias."""
    np.random.seed(0)
    nn = logreg.NeuralNetwork(2, 1, 0.001)
    assert nn.W.shape == (2, 1)
    assert nn.b == pytest.approx([0.0])


def test_forward_values(logreg):
    """forward computes sigmoid(x W + b) for every row of x."""
    nn = make_network(logreg, [1.0, -2.0], 0.5)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    expected = 1.0 / (1.0 + np.exp(-np.array([[0.5], [-0.5], [2.5]])))
    out = nn.forward(x)
    assert out.shape == (3, 1)
    assert out == pytest.approx(expected)


def test_cross_entropy_hand_values(logreg):
    """Cross-entropy is the mean of -y ln(p) - (1 - y) ln(1 - p) over the batch."""
    nn = make_network(logreg, [0.0, 0.0], 0.0)
    Y = np.array([[1.0], [0.0]])
    P = np.array([[0.8], [0.4]])
    assert nn.cross_entropy(Y, P) == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2.0)


def test_cross_entropy_of_uninformed_prediction(logreg):
    """Predicting 0.5 everywhere costs ln 2."""
    nn = make_network(logreg, [0.0, 0.0], 0.0)
    Y = np.array([[1.0], [0.0], [1.0], [1.0]])
    assert nn.cross_entropy(Y, np.full((4, 1), 0.5)) == pytest.approx(np.log(2.0))


def test_backward_matches_finite_differences(logreg):
    """The analytic gradients equal central finite differences of the loss."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    Y = (rng.random((8, 1)) > 0.5).astype(float)
    W0 = rng.normal(size=(2, 1))
    b0 = 0.3
    nn = make_network(logreg, W0, b0)
    dW, db = nn.backward(X, Y, nn.forward(X))
    assert dW.shape == (2, 1)

    def loss(W, b):
        probe = make_network(logreg, W, b)
        return probe.cross_entropy(Y, probe.forward(X))

    eps = 1e-6
    numeric_dW = np.zeros((2, 1))
    for k in range(2):
        step = np.zeros((2, 1))
        step[k] = eps
        numeric_dW[k] = (loss(W0 + step, b0) - loss(W0 - step, b0)) / (2 * eps)
    numeric_db = (loss(W0, b0 + eps) - loss(W0, b0 - eps)) / (2 * eps)

    assert dW == pytest.approx(numeric_dW, abs=1e-7)
    assert db == pytest.approx(numeric_db, abs=1e-7)


def test_update_weights_takes_gradient_step(logreg):
    """update_weights moves W and b against the gradient by the learning rate."""
    nn = make_network(logreg, [1.0, 2.0], 0.5, learning_rate=0.1)
    nn.update_weights(np.array([[10.0], [-20.0]]), 5.0)
    assert nn.W == pytest.approx(np.array([[0.0], [4.0]]))
    assert nn.b == pytest.approx([0.0])


def test_predict_thresholds_at_one_half(logreg):
    """predict returns 1 where the probability exceeds 0.5 and 0 elsewhere."""
    nn = make_network(logreg, [1.0, 0.0], 0.0)
    x = np.array([[2.0, 0.0], [-2.0, 0.0], [0.5, 9.0]])
    assert nn.predict(x).ravel().tolist() == [1.0, 0.0, 1.0]


def test_load_and_normalize_data(logreg):
    """The data set has 100 examples; after normalization every feature has mean 0, std 1."""
    X_all, Y_all = logreg.load_data()
    assert X_all.shape == (100, 2)
    assert Y_all.shape == (100, 1)
    assert set(np.unique(Y_all)) == {0.0, 1.0}
    X_norm, m, s = logreg.normalize(X_all)
    assert m.shape == (2,) and s.shape == (2,)
    assert X_norm.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-5)
    assert X_norm.std(axis=0, ddof=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_short_training_reduces_loss(logreg):
    """A few epochs with a fixed seed lower the epoch loss monotonically."""
    X_all, Y_all = logreg.load_data()
    X_all, _, _ = logreg.normalize(X_all)
    np.random.seed(0)
    nn = logreg.NeuralNetwork(2, 1, 0.001)
    losses = logreg.train(nn, X_all, Y_all, batch_size=8, num_epochs=5, verbose=False)
    assert len(losses) == 5
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))


def test_training_is_deterministic_with_seed(logreg):
    """Two runs with the same seed give identical weights."""
    X_all, Y_all = logreg.load_data()
    X_all, _, _ = logreg.normalize(X_all)
    weights = []
    for _ in range(2):
        np.random.seed(5)
        nn = logreg.NeuralNetwork(2, 1, 0.001)
        logreg.train(nn, X_all, Y_all, batch_size=8, num_epochs=2, verbose=False)
        weights.append(np.concatenate([nn.W.ravel(), nn.b.ravel()]))
    assert weights[0] == pytest.approx(weights[1], abs=0)


@pytest.mark.slow
def test_full_training_accuracy_and_example_probability(logreg):
    """500 epochs, seed 0: training accuracy 0.90 and P([45, 85]) within [0.62, 0.67]."""
    X_all, Y_all = logreg.load_data()
    X_all, m, s = logreg.normalize(X_all)
    np.random.seed(0)
    nn = logreg.NeuralNetwork(logreg.num_features, logreg.output_depth, logreg.learning_rate)
    losses = logreg.train(nn, X_all, Y_all, logreg.batch_size, logreg.num_epochs, verbose=False)
    assert len(losses) == 500
    assert losses[-1] < 0.25
    accuracy = np.mean(nn.predict(X_all) == Y_all)
    assert accuracy == pytest.approx(0.90)
    probability = nn.forward((np.array([[45.0, 85.0]]) - m) / s).item()
    assert 0.62 <= probability <= 0.67


@pytest.mark.slow
def test_main_prints_probability_and_accuracy(logreg, monkeypatch, capsys):
    """The script itself (answering 'no' to the plot prompt) prints the same results."""
    pytest.importorskip("sklearn")
    monkeypatch.setattr("builtins.input", lambda prompt="": "0")
    np.random.seed(0)
    logreg.main()
    out = capsys.readouterr().out
    assert out.count("epoch_loss") == 500
    probability = float(re.search(r"Probability =\s+([0-9.]+)", out).group(1))
    accuracy = float(re.search(r"accuracy =\s+([0-9.]+)", out).group(1))
    assert 0.62 <= probability <= 0.67
    assert accuracy == pytest.approx(0.90)
    assert "Logistic Reggresion using sklearn ready classifier:" in out


# ===================================================================== 4.3

@pytest.fixture(scope="module")
def tf():
    tensorflow = pytest.importorskip("tensorflow")
    tensorflow.keras.utils.set_random_seed(0)
    return tensorflow


@pytest.fixture(scope="module")
def fashion(tf):
    return load_exercise(FASHION_DIR / "fashion.py")


@pytest.fixture(scope="module")
def fashion_cnn(tf):
    return load_exercise(FASHION_DIR / "fashion_cnn.py")


def test_cnn_parameter_count_and_output_shape(fashion_cnn):
    """The CNN of the assignment has exactly 373,538 parameters and 10 outputs."""
    model = fashion_cnn.build_model()
    assert model.count_params() == 373538
    assert model.output_shape == (None, 10)


def test_cnn_layer_structure(fashion_cnn):
    """5 convolutions, 6 batch normalizations, 3 poolings, dropouts 0.2/0.3/0.4/0.5."""
    model = fashion_cnn.build_model()
    kinds = [type(layer).__name__ for layer in model.layers]
    assert kinds.count("Conv2D") == 5
    assert kinds.count("BatchNormalization") == 6
    assert kinds.count("MaxPooling2D") == 3
    rates = [layer.rate for layer in model.layers if type(layer).__name__ == "Dropout"]
    assert rates == pytest.approx([0.2, 0.3, 0.4, 0.5])
    filters = [layer.filters for layer in model.layers if type(layer).__name__ == "Conv2D"]
    assert filters == [32, 32, 64, 64, 128]


def test_dense_model_output_shape(fashion):
    """The dense model maps 28 x 28 images to 10 logits (101,770 parameters)."""
    model = fashion.build_model()
    assert model.output_shape == (None, 10)
    assert model.count_params() == 784 * 128 + 128 + 128 * 10 + 10


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop", "nadam", "adamax", "ftrl"])
def test_dense_model_compiles_with_every_optimizer(fashion, optimizer):
    """Every optimizer of the comparison is accepted by build_model."""
    assert optimizer in fashion.OPTIMIZERS
    model = fashion.build_model(optimizer)
    assert model.optimizer is not None


def test_dense_forward_pass_gives_finite_logits(fashion):
    """A random batch of 4 images gives a finite (4, 10) array of logits."""
    images = np.random.default_rng(0).random((4, 28, 28)).astype("float32")
    logits = fashion.build_model().predict(images, verbose=0)
    assert logits.shape == (4, 10)
    assert np.all(np.isfinite(logits))


def test_cnn_forward_pass_gives_finite_logits(fashion_cnn):
    """A random batch of 4 images gives a finite (4, 10) array of logits."""
    images = np.random.default_rng(0).random((4, 28, 28, 1)).astype("float32")
    logits = fashion_cnn.build_model().predict(images, verbose=0)
    assert logits.shape == (4, 10)
    assert np.all(np.isfinite(logits))


@pytest.mark.slow
def test_dense_training_step_runs(fashion):
    """One epoch on 32 random images runs and reports a finite loss."""
    rng = np.random.default_rng(0)
    images = rng.random((32, 28, 28)).astype("float32")
    labels = rng.integers(0, 10, size=32)
    history = fashion.build_model().fit(images, labels, epochs=1, batch_size=32, verbose=0)
    assert np.isfinite(history.history["loss"][0])
    assert 0.0 <= history.history["accuracy"][0] <= 1.0


@pytest.mark.slow
def test_cnn_training_step_runs(fashion_cnn):
    """One epoch on 32 random images runs and reports a finite loss."""
    rng = np.random.default_rng(0)
    images = rng.random((32, 28, 28, 1)).astype("float32")
    labels = rng.integers(0, 10, size=32)
    history = fashion_cnn.build_model().fit(images, labels, epochs=1, batch_size=32, verbose=0)
    assert np.isfinite(history.history["loss"][0])
    assert 0.0 <= history.history["accuracy"][0] <= 1.0


def test_plot_helpers_draw_without_window(tf):
    """The plotting helpers of plots.py draw their figures with the Agg backend."""
    import matplotlib.pyplot as plt
    plots = load_exercise(FASHION_DIR / "plots.py")
    rng = np.random.default_rng(0)
    images = rng.random((25, 28, 28))
    labels = rng.integers(0, 10, size=25)
    predictions = rng.random((25, 10))
    predictions /= predictions.sum(axis=1, keepdims=True)
    names = [str(k) for k in range(10)]
    try:
        plots.plot_some_data(images, labels, names)
        assert len(plt.gcf().axes) == 25
        plots.plot_some_predictions(images, labels, predictions, names, num_rows=2, num_cols=2)
        assert len(plt.gcf().axes) == 8
    finally:
        plt.close("all")

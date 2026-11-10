from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Question 2: comparison of the optimization algorithms.
# Run with:  python fashion.py compare
# Every optimizer is trained from scratch on the same architecture for
# COMPARE_EPOCHS epochs and the training / validation accuracies are plotted.
OPTIMIZERS = ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'ftrl']
COMPARE_EPOCHS = 50


# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.

def build_model(optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def compare_optimizers(train_images, train_labels, test_images, test_labels):
    results = {}
    plt.figure(figsize=(15, 8))
    for k, optimizer in enumerate(OPTIMIZERS):
        print('\nOptimizer:', optimizer)
        model = build_model(optimizer)
        history = model.fit(train_images, train_labels, epochs=COMPARE_EPOCHS,
                            validation_data=(test_images, test_labels))
        results[optimizer] = (history.history['accuracy'][-1], history.history['val_accuracy'][-1])

        plt.subplot(2, 3, k+1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.title(optimizer)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

    print('\nOptimizer comparison after %d epochs:' % COMPARE_EPOCHS)
    for optimizer in OPTIMIZERS:
        print('%-8s accuracy: %.4f  val_accuracy: %.4f' % ((optimizer,) + results[optimizer]))
    best = max(OPTIMIZERS, key=lambda name: results[name][1])
    print('Best validation accuracy:', best)

    plt.tight_layout()
    plt.show()


def main():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Scale these values to a range of 0 to 1 before feeding them to the neural network model.
    # To do so, divide the values by 255.
    # It's important that the training set and the testing set be preprocessed in the same way
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    plot_some_data(train_images, train_labels, class_names)

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_optimizers(train_images, train_labels, test_images, test_labels)
        return

    # Question 1: the given architecture with the adam optimizer
    model = build_model('adam')

    #Train the model
    # Training the neural network model requires the following steps:

    #   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
    #   2. The model learns to associate images and labels.
    #   3. You ask the model to make predictions about a test set—in this example, the test_images array.
    #   4. Verify that the predictions match the labels from the test_labels array.

    history = model.fit(train_images, train_labels, epochs=400, validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    # Make predictions
    # With the model trained, you can use it to make predictions about some images.
    # The model's linear outputs, logits.
    # Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)


if __name__ == "__main__":
    main()

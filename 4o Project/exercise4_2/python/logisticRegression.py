#!/usr/bin/python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / 'exam_scores_data1.txt'

# Define the model
num_features = 2
output_depth = 1
batch_size = 8
learning_rate = 0.001
num_epochs = 500


def load_data(filename=DATA_FILE):
    """Read the examples: returns X (num_examples, 2) and Y (num_examples, 1)."""
    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()

    input_list = []
    target_list = []

    for line in lines:
        fields = line.rstrip().split(',')
        input_list.append([float(fields[0]), float(fields[1])])
        target_list.append([float(fields[2])])

    # Matrix of input features. shape = (num_examples, num_features)
    X_all = np.array(input_list, dtype=np.float32)
    # Vector of target values. shape = (num_examples, 1)
    Y_all = np.array(target_list, dtype=np.float32)
    return X_all, Y_all


def normalize(X_all):
    """Normalize every feature separately. Returns (X normalized, mean, std)."""
    m = np.mean(X_all, axis=0)          # per feature
    s = np.std(X_all, axis=0, ddof=1)   # sample std, same as matlab std()
    return (X_all-m)/s, m, s


# sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    def __init__(self, input_depth, output_depth, learning_rate):
        self.W = np.sqrt(2.0/(input_depth + output_depth)) * \
            np.random.randn(input_depth, output_depth).astype(np.float32)
        self.b = np.zeros((output_depth, ), dtype=np.float32)
        self.learning_rate = learning_rate

    # forward propagation using sigmoid fuction
    def forward(self, x):
        tmp = np.dot(x, self.W) + self.b
        output = sigmoid(tmp)  # calling activation function
        return output

    # backward propagation using gradient descent
    def backward(self, X, Y, Y_predicted):
        m = X.shape[0]

        dZ = Y_predicted - Y

        dW = 1/m * np.dot(X.T, dZ)  # same shape as W (2 x 1)
        db = 1/m * np.sum(dZ)

        return dW, db

    # cost calculation using cross entropy
    def cross_entropy(self, Y, Y_predicted):
        self.loss = -np.mean(np.multiply(Y, np.log(Y_predicted)) + np.multiply((1-Y), np.log(1-Y_predicted)))

        return self.loss

    # update parameters
    def update_weights(self, d_CE_d_W, d_CE_d_b):
        self.W = self.W - self.learning_rate * d_CE_d_W
        self.b = self.b - self.learning_rate * d_CE_d_b

    # make prediction
    def predict(self, x):
        return (self.forward(x) > 0.5).astype(np.float32)


def train(nn, X_all, Y_all, batch_size, num_epochs, verbose=True):
    """Train nn with a sliding window of length batch_size and shift 1.
    Returns the list with the mean loss of every epoch."""
    N = X_all.shape[0]  # Number of examples
    num_batches = N - batch_size + 1
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(num_batches):  # Sliding window of length = batch_size and shift = 1
            X = X_all[i:i+batch_size, :]
            Y = Y_all[i:i+batch_size, :]

            Y_predicted = nn.forward(X)
            batch_loss = nn.cross_entropy(Y, Y_predicted)
            epoch_loss += batch_loss

            d_CE_d_W, d_CE_d_b = nn.backward(X, Y, Y_predicted)
            nn.update_weights(d_CE_d_W, d_CE_d_b)
        epoch_loss /= num_batches
        losses.append(epoch_loss)
        if verbose:
            print('epoch_loss = ', epoch_loss)
    return losses


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import DataConversionWarning

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Load data and normalize input data
    X_all, Y_all = load_data()
    X_all, m, s = normalize(X_all)

    nn = NeuralNetwork(num_features, output_depth, learning_rate)
    train(nn, X_all, Y_all, batch_size, num_epochs)

    # Using the trained model to predict the probabilities of some examples and the compute the accuracy

    # Predict the normalized example [45, 85]
    example = (np.array([[45, 85]], dtype=np.float32) - m)/s
    print('\nPredicting the probabilities of example [45, 85]\n')

    Pr_example_2 = nn.forward(example)
    print("Probability = ", Pr_example_2.item())
    print("\n")

    # Predict the accuracy of the training examples
    Y_predicted = nn.predict(X_all)

    correct = 0
    for t in range(Y_all.shape[0]):
        if Y_all[t] == Y_predicted[t]:
            correct += 1

    accuracy_np = correct / np.size(Y_all)

    print('accuracy = ', accuracy_np)

    ####################################################################
    # comparison of the original dataset and the nn classification #####
    ####################################################################
    print("Do you want to print the dataset?\n")
    image = input("Press 1 for yes, or anything for no:")

    if image == '1':
        ones = np.where(Y_all.ravel() == 1)[0]
        zeros = np.where(Y_all.ravel() == 0)[0]
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.xticks([]), plt.yticks([])
        plt.title("Original dataset")
        plt.scatter(X_all[ones, 0], X_all[ones, 1])
        plt.scatter(X_all[zeros, 0], X_all[zeros, 1])
        plt.xlabel("Grade 1")
        plt.ylabel("Grade 2")

        plt.subplot(2, 2, 2)
        plt.xticks([]), plt.yticks([])
        ones = np.where(Y_predicted.ravel() == 1)[0]
        zeros = np.where(Y_predicted.ravel() == 0)[0]
        plt.title("NeuralNetwork prediction")
        plt.scatter(X_all[ones, 0], X_all[ones, 1])
        plt.scatter(X_all[zeros, 0], X_all[zeros, 1])
        plt.xlabel("Grade 1")
        plt.ylabel("Grade 2")
        plt.show()

    ###########################################################################
    # sklearn logistic reggression for comparison #############################
    ###########################################################################
    print("\nLogistic Reggresion using sklearn ready classifier:")
    X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all.ravel(), test_size=0.20)
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    model_prediction = model.predict(X_test)
    print(accuracy_score(y_test, model_prediction))


if __name__ == "__main__":
    main()

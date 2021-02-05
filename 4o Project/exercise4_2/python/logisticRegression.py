#!/usr/bin/python3
# -*- coding: utf-8 -*-

# from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning

simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Load data
fid = open('exam_scores_data1.txt', 'r')
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
# Vector of target values. shape = (num_examples, )
Y_all = np.array(target_list, dtype=np.float32)

N = X_all.shape[0]  # Number of examples

# Normalize input data
# your code here
m = np.mean(X_all)
s = np.std(X_all)

X_all = (X_all-m)/s

# Define the model
num_features = 2
output_depth = 1
batch_size = 8
learning_rate = 0.001

# sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class NeuralNetwork:
    def __init__(self, input_depth, output_depth, learning_rate):
        self.W = np.sqrt(2.0/(input_depth + output_depth)) * \
            np.random.rand(input_depth, output_depth).astype(np.float32)
        self.b = np.zeros((output_depth, ), dtype=np.float32)
        self.learning_rate = learning_rate

    # forward propagation using sigmoid fuction
    def forward(self, x):
        tmp = np.mat(x) * np.mat(self.W) + self.b
        output = sigmoid(tmp) #calling activation function
        return output

    # backward propagation using gradient descent 
    def backward(self, X, Y, Y_predicted):

        # your code here
        m = X.shape[0]

        dZ = Y_predicted - Y

        dW = 1/m * np.dot(dZ.T,X)
        db = 1/m * np.sum(dZ)

        return dW , db

    # cost calculation using cross entropy
    def cross_entropy(self, Y, Y_predicted):
        # your code here
        self.loss = -np.mean(np.multiply(Y, np.log(Y_predicted)) + np.multiply((1-Y), np.log(1-Y_predicted)))

        return self.loss

    # update parameters
    def update_weights(self, d_CE_d_W, d_CE_d_b):
        tmp = np.dot(self.learning_rate,d_CE_d_W)
        tmp = tmp.reshape(2,1)
        self.W = self.W - tmp
        self.b = self.b - self.learning_rate * d_CE_d_b
    # make prediction
    def predict(self, x):
    	prediction = nn.forward(x)
    	for i in range(np.size(prediction)):
        	if prediction[i] > 0.5:
        		prediction[i] = int(1)
        	else:
        		prediction[i] = int(0)
    	return prediction



nn = NeuralNetwork(num_features, output_depth, learning_rate)
num_epochs = 55
num_batches = N - batch_size + 1

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
    print('epoch_loss = ', epoch_loss)

# Using the trained model to predict the probabilities of some examples and the compute the accuracy

# Predict the normalized example [45, 85]
example = (np.array([[45, 85]], dtype=np.float32) - m)/s
print('\nPredicting the probabilities of example [45, 85]\n')

Pr_example_2 = nn.forward(example)
print("Probability = ",Pr_example_2)
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
	ones = np.argwhere(Y_all==1)
	zeros = np.argwhere(Y_all==0)
	plt.figure()
	plt.subplot(2,2,1)
	xticks([]), yticks([])
	plt.title("Original dataset")
	plt.scatter(X_all[ones,0],X_all[ones,1])
	plt.scatter(X_all[zeros,0],X_all[zeros,1])
	plt.xlabel("Grade 1")
	plt.ylabel("Grade 2")

	plt.subplot(2,2,2)
	xticks([]), yticks([])
	ones = np.argwhere(Y_predicted==1)
	zeros = np.argwhere(Y_predicted==0)
	plt.title("NeuralNetwork prediction")
	plt.scatter(X_all[ones,0],X_all[ones,1])
	plt.scatter(X_all[zeros,0],X_all[zeros,1])
	plt.xlabel("Grade 1")
	plt.ylabel("Grade 2")
	plt.show()

###########################################################################
# sklearn logistic reggression for comparison #############################
###########################################################################
print("\nLogistic Reggresion using sklearn ready classifier:")
X_train,X_test,y_train,y_test = train_test_split(X_all,Y_all.ravel(),test_size=0.20)
model = LogisticRegression(random_state=0).fit(X_train,y_train)
model_prediction = model.predict(X_test)
print(accuracy_score(y_test,model_prediction))
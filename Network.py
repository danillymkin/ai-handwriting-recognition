import random

import numpy as np
import sys
import matplotlib.pyplot as plt

from constants import LEARNING_RATE, MAX_EPOCHS, SIGMOID_SLOPE_FACTOR
from utils import sigmoid, soft_max, to_full, sparse_cross_entropy, sigmoid_deriv


class Network:
    def __init__(self, structure):
        if len(structure) != 3:
            sys.exit(f"Invalid structure: {len(structure)} layers are specified instead of 3")

        self.INPUT_NEURONS_COUNT = structure[0]
        self.HIDE_NEURONS_COUNT = structure[1]
        self.OUTPUT_NEURONS_COUNT = structure[2]

        self.W1 = np.random.randn(self.INPUT_NEURONS_COUNT, self.HIDE_NEURONS_COUNT)
        self.b1 = np.random.randn(self.HIDE_NEURONS_COUNT)
        self.W2 = np.random.randn(self.HIDE_NEURONS_COUNT, self.OUTPUT_NEURONS_COUNT)
        self.b2 = np.random.randn(self.OUTPUT_NEURONS_COUNT)

        self.h1 = None
        self.t1 = None
        self.h2 = None
        self.t2 = None

        self.dE_dW1 = None
        self.dE_db1 = None
        self.dE_dW2 = None
        self.dE_db2 = None

        self.loss = []

    def learn(self, dataset):
        for epoch in range(MAX_EPOCHS):
            random.shuffle(dataset)

            for i in range(len(dataset)):
                x, y = dataset[i]

                z, e, = self.forward(x, y)
                self.backward(x, y, z)
                self.update_weights()

                self.loss.append(e)

    def predict(self, x):
        t1 = x @ self.W1 + self.b1
        h1 = sigmoid(t1, SIGMOID_SLOPE_FACTOR[0])
        t2 = h1 @ self.W2 + self.b2
        h2 = sigmoid(t2, SIGMOID_SLOPE_FACTOR[1])
        z = soft_max(h2)
        return z

    def forward(self, x, y):
        self.t1 = x @ self.W1 + self.b1
        self.h1 = sigmoid(self.t1, SIGMOID_SLOPE_FACTOR[0])
        self.t2 = self.h1 @ self.W2 + self.b2
        self.h2 = sigmoid(self.t2, SIGMOID_SLOPE_FACTOR[1])
        z = soft_max(self.h2)
        e = sparse_cross_entropy(z, y)
        return z, e

    def backward(self, x, y, z):
        y_full = to_full(y, self.OUTPUT_NEURONS_COUNT)
        dE_dt2 = z - y_full
        self.dE_dW2 = self.h1.T @ dE_dt2
        self.dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ self.W2.T
        dE_dt1 = dE_dh1 * sigmoid_deriv(self.t1, SIGMOID_SLOPE_FACTOR[0])
        self.dE_dW1 = x.T @ dE_dt1
        self.dE_db1 = dE_dt1

    def update_weights(self):
        self.W1 = self.W1 - LEARNING_RATE * self.dE_dW1
        self.b1 = self.b1 - LEARNING_RATE * self.dE_db1
        self.W2 = self.W2 - LEARNING_RATE * self.dE_dW2
        self.b2 = self.b2 - LEARNING_RATE * self.dE_db2

    def calc_accuracy(self, dataset):
        correct = 0
        for x, y in dataset:
            z = self.predict(x)
            y_predict = np.argmax(z)
            if y_predict == y:
                correct += 1
        accuracy = correct / len(dataset)
        return accuracy

    def show_error_graph(self):
        plt.plot(self.loss)
        plt.title('Loss')
        plt.show()

    def test(self, x):
        probs = self.predict(x)
        pred_class = np.argmax(probs)
        class_names = ['0', '1', 'Ж', 'Ф']
        print('Predicted class:', class_names[pred_class])

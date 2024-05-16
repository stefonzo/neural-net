#!/usr/bin/env python

import numpy as np
from random import shuffle, sample
from sklearn.datasets import load_iris


class layer:
    # inputs: number of inputs
    # outputs: number of outputs
    # activation: tuple, containing an activation function
    #   and its derivative
    def __init__(self, inputs, outputs, activation):
        self.weights = np.random.rand(outputs, inputs)
        self.biases = np.random.rand(outputs)
        self.activation = activation
        self.errors = []
        self.activations = []
        self.z = None

    # compute the network forward
    def forward(self, input):
        self.z = np.matmul(self.weights, input) + self.biases
        self.activations.append(self.activation[0](self.z))
        # print(input)
        # print("v")
        # print(self.z)
        # print("v")
        # print(self.activations[-1])
        # print()
        return self.activations[-1]

    # backprop
    def backward(self, following):
        self.errors.append(following * self.activation[1](self.z))
        return np.matmul(np.transpose(self.weights), self.errors[-1])

    def train(self, rate, input):
        s = sum(map(
            lambda x: np.outer(x[0], x[1]),
            zip(self.errors, input)
        ))
        self.weights -= (s * (rate/len(self.errors)))
        self.biases -= (rate/len(self.errors)) * sum(self.errors)

        result = self.activations
        self.errors = []
        self.activations = []
        return result

    def print(self):
        print(self.weights)
        print(self.biases)


class network:
    def __init__(self, layers):
        self.layers = layers

    def compute(self, input):
        a = input
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def train_batch(self, pairs, cost):
        for (input, expected) in pairs:
            following = cost[1](expected, self.compute(input))
            for layer in reversed(self.layers):
                following = layer.backward(following)

        a = [ x[0] for x in pairs ]
        for layer in self.layers:
            a = layer.train(0.1, a)

    def train(self, train, test, epochs, batch_size, cost):
        for _ in range(epochs):
            batch = sample(train, batch_size)
            self.train_batch(batch, cost)
            print("error: ", self.test(test, cost))
            # for layer in self.layers:
            #     layer.print()
            # print()

    def test(self, data, cost):
        costs = [ 
            cost[0](expected, self.compute(input)) 
            for (input, expected) 
            in data 
        ]
        return sum(costs) / len(costs)


def sig(x): 
    return 1/(1 + np.exp(-x))
sigmoid = (
    sig,
    lambda x: sig(x)*(1 - sig(x))
)

quadcost = (
    lambda expected, actual: 0.5 * (np.abs(expected - actual)**2),
    lambda expected, actual: (actual - expected)
)



data, target = load_iris(return_X_y=True)
def one_hot(arr):
    l = max(arr)+1
    result = []
    for x in arr:
        y = np.zeros(l)
        y[x] = 1
        result.append(y)
    return result

pairs = [ x for x in zip(data, one_hot(target)) ]
shuffle(pairs)
train = pairs[0:130]
test = pairs[130:]


n = network([
    layer(4, 8, sigmoid),
    layer(8, 6, sigmoid),
    layer(6, 3, sigmoid),
])


n.train(train, test, 3000, 20, quadcost)

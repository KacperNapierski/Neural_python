import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # n_inputs = number of inputs in a batch
        # n_neurons = number of neurons in new layer we want
        # matrix inputs x neurons so ther e will be no need to transpose in forward
        self.weights = 0.1* np.random.randn(n_inputs, n_neurons) #0.1 to have values below 1
        self.biases = np.zeros((1, n_neurons)) # the tuple is the shape -> generates 1xNEURONS vector
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X , y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3) # inputs is x,y coordinates of poin on axis
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3) # output is 3 as we have 3 classes only
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])




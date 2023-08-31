import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # n_inputs = number of inputs in a batch
        # n_neurons = number of neurons in new layer we want
        # matrix inputs x neurons so ther e will be no need to transpose in forward
        self.weights = 0.1* np.random.randn(n_inputs, n_neurons) #0.1 to have values below 1
        self.biases = np.zeros((1, n_neurons)) # the tuple is the shape -> generates 1xNEURONS vector
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
 
layer1.forward(X)
print(layer1.output)
print('###')
layer2.forward(layer1.output)
print(layer2.output)
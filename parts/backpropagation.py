import numpy as np

#forward data
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

bias = np.array([[2.0, 3.0, 0.5]])
print(weights.T)
print("########")
layer_output= np.dot(inputs, weights) + bias
relu_output = np.maximum(0, layer_output)
print(layer_output)
print("########")
print(relu_output)
print("#################")
drelu = relu_output.copy()
print(drelu)

drelu[layer_output <= 0] = 0
print("########")
print(drelu)

dinputs = np.dot(drelu, weights.T)
print(dinputs)
print("########")
dweights = np.dot(inputs.T, drelu)
print(dweights)
print("########")
dbiases = np.sum(drelu, axis=0, keepdims=True) #sum in columns
print(dbiases)

weights += -0.001*dweights
bias += -0.001*dbiases
print("#################")
print(weights)
print(bias)
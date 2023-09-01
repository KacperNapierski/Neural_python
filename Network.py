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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self,dvalues):
        self.dweight = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues,axis=0, keepdims= True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backwards(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y): #output is model output, y- intendet target values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_prediction, y_true): #y_pred values from neural network, y_true target trening values
        samples = len(y_prediction)
        #clip to prevent loss going to inf while confidence is 0 -> limits the range of values
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1: #someone inputed scalar not one-hot encoded
            correct_confidences = y_prediction_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_prediction_clipped*y_true, axis=1)
        negative_log_probability = -np.log(correct_confidences)
        return negative_log_probability
    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = - y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples)] -= 1
        self.dinputs = self.dinputs / samples

#X , y = spiral_data(samples=100, classes=3)
#
#dense1 = Layer_Dense(2, 3) # inputs is x,y coordinates of poin on axis
#activation1 = Activation_ReLU()
#
#dense2 = Layer_Dense(3, 3) # output is 3 as we have 3 classes only
#activation2 = Activation_Softmax()
#
#dense1.forward(X)
#print(dense1.output)
#activation1.forward(dense1.output)
#print(activation1.forward(dense1.output))
#
#dense2.forward(activation1.output)
#activation2.forward(dense2.output)
#
#print(activation2.output[:5])
#
#loss_function = Loss_CategoricalCrossEntropy()
#loss = loss_function.calculate(activation2.output, y)
#
#prediction = np.argmax(activation2.output, axis=1)
#if y.shape == 2:
#    y = np.argmax(y, axis=1)
#accuracy = np.mean(prediction == y)    
#
#print(f"Loss {loss}")
#print(accuracy)

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_target = np.array([0, 1, 1])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
softmax_loss.backward(softmax_output, class_target)
dvalues1 = softmax_loss.dinputs
print("Gradients: loss + activation combination")
print(dvalues1)

activation = Activation_Softmax()
activation.output = softmax_output
loss = Loss_CategoricalCrossEntropy()
loss.backward(softmax_output, class_target)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs
print("Gradients: loss and activation separetly")
print(dvalues2)
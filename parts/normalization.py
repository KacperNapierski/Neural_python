#from math import e
import numpy as np 

#layer_output = [4.8,1.21,2.385]

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [ 1.41, 1.051, 0.026]]

# CLEAR PYTHON METHOD
#exp_values = []
#
#for output in layer_output:
#    exp_values.append(e**output)
#
#print(exp_values)
#
#normalized_base = sum(exp_values)
#normalized_values = []
#
#for value in exp_values:
#    normalized_values.append(value/normalized_base)
#
#print(normalized_values)
#print(sum(normalized_values))
 
#NUMPY METHOD

#exp_values = np.exp(layer_output)
#normalization_values = exp_values / np.sum(exp_values)

#print(normalization_values)
#print(np.sum(normalization_values))

#BATCH INPUT

exp_values = np.exp(layer_output)
normalization_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(normalization_values)

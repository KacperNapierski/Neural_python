neuron_inputs = [1, 2, 3, 2.5] #outputs form 4 neurons from previous layer
weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]
output = []

for j in range(3):
    output.insert(j,0)
    for i in range(len(neuron_inputs)):
        output[j]+= neuron_inputs[i] * weights[j][i]

    output[j] += bias[j]

print(output)
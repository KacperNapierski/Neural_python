import math
import numpy as np

#softmax_output = [0.7, 0.1, 0.2]
#target_output = [1, 0, 0]
#loss = 0
#
#for i in range(len(softmax_output)):
#    loss += -(math.log(softmax_output[i])*target_output[i])
#
#print(loss)

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])

predictions = np.argmax(softmax_output, axis=1)
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions == class_targets)
print(f"accuracy = {accuracy}")

negative_log = -np.log(softmax_output[range(len(softmax_output)),class_targets])
average_loss = np.mean(negative_log)
print(average_loss)
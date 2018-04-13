import neural_net as nn
import numpy as np

LEARNING_RATE = 0.8
ACTIVATION = nn.Sigmoid
RANDOM_WEIGHTS = True
LAYERS = (2,)

INPUTS = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
OUTPUTS = np.array([[0, 1, 1, 0]])

a = nn.Net(INPUTS, OUTPUTS, layers=(2,), mid_layer_activation=ACTIVATION, output_layer_activation=ACTIVATION,
           lr=LEARNING_RATE, random=RANDOM_WEIGHTS)

print("Initial Weights: ")
for i in range(len(a.weights)):
    print("Layer {}:\n {}".format(i, a.weights[i]))

a.train(n_iter=100)

result = a.forward()

print("Forward pass results: ")
for i in range(a.inputs.shape[1]):
    print("{} {} : {}".format(a.inputs.T[i][0], a.inputs.T[i][1], result[0].T[i]))

print("Weights: ")
for i in range(len(a.weights)):
    print("Layer {}:\n {}".format(i, a.weights[i]))

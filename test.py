import neural_net as nn
import numpy as np

LEARNING_RATE = 0.8
ACTIVATION = nn.Sigmoid
RANDOM_WEIGHTS = True
LOSS_FN = nn.CrossEntropyLoss
LAYERS = (2, 5, 5, 5, 1)

INPUTS = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
OUTPUTS = np.array([[0, 1, 1, 0]])

a = nn.Net(layers=LAYERS, activation=ACTIVATION, loss=LOSS_FN, lr=LEARNING_RATE, random=RANDOM_WEIGHTS)

print("Initial Weights: ")
for i in range(len(a.weights)):
    print("Layer {}:\n {}\n {}".format(i, a.weights[i], a.biases[i]))

a.fit(INPUTS, OUTPUTS, n_iter=1000)

result = a.predict(INPUTS)

print("Forward pass results: ")
for i in range(INPUTS.shape[1]):
    print("{} {} : {}".format(INPUTS.T[i][0], INPUTS.T[i][1], result[0].T[i]))

print("Weights: ")
for i in range(len(a.weights)):
    print("Layer {}:\n {}\n {}".format(i, a.weights[i], a.biases[i]))

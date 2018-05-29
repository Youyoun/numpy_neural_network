from sklearn.datasets import load_iris
import neural_net as nn
import numpy as np

LEARNING_RATE = 0.01
ACTIVATION = nn.Softmax
RANDOM_WEIGHTS = True
LOSS_FN = nn.CrossEntropyLoss
LAYERS = (4, 20, 10, 1)

DATASET = load_iris()
INPUTS = DATASET.data.T
OUTPUTS = np.array([DATASET.target])
print(OUTPUTS.shape)

a = nn.Net(layers=LAYERS, activation=ACTIVATION, loss=LOSS_FN, lr=LEARNING_RATE, random=RANDOM_WEIGHTS)

a.fit(INPUTS, OUTPUTS)

result = a.predict(INPUTS)

print("Accuracy: {}".format("Not Implemented"))

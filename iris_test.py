from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import neural_net as nn
import numpy as np

LEARNING_RATE = 0.001
ACTIVATION = nn.Softmax
RANDOM_WEIGHTS = True
LOSS_FN = nn.CrossEntropyLoss
LAYERS = (4, 8, 3)

DATASET = load_iris()
INPUTS = DATASET.data
OUTPUTS = DATASET.target
X_train, X_test, Y_train, Y_test = train_test_split(INPUTS, OUTPUTS, test_size=0.2, random_state=42)

Y_train_softmax = []
for i in range(len(Y_train)):
    Y_train_softmax.append(np.zeros(3))
    Y_train_softmax[i][Y_train[i]] = 1
Y_train_softmax = np.array(Y_train_softmax)
print(Y_train.shape, Y_train_softmax.shape)

a = nn.Net(layers=LAYERS, activation=ACTIVATION, loss=LOSS_FN, lr=LEARNING_RATE, random=RANDOM_WEIGHTS)

a.fit(X_train, Y_train_softmax, n_iter=10000)

result = a.predict(X_test)
result = [np.argmax(e) for e in result]
print(result)
print("Accuracy: {}".format(np.sum(result == Y_test)))

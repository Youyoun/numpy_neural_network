from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import neural_net as nn
import numpy as np
import activation as act
import loss

LEARNING_RATE = 0.001
ACTIVATION = act.ReLU
LOSS_FN = loss.CrossEntropyLoss
LAYERS = (10,)

DATASET = load_iris()
INPUTS = DATASET.data
OUTPUTS = DATASET.target
X_train, X_test, Y_train, Y_test = train_test_split(INPUTS, OUTPUTS, test_size=0.2, random_state=42)

a = nn.Net(hidden_layers=LAYERS, activation=ACTIVATION, loss_function=LOSS_FN, lr=LEARNING_RATE)#, descent="stochastic")

a.fit(X_train, Y_train, epochs=100000)

result = a.predict(X_test)
print(result)
print("Accuracy: {}".format(np.sum(result == Y_test)))

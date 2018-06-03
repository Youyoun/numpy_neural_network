from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import neural_net as nn
import numpy as np
import activation as act
import loss

LEARNING_RATE = 0.04
ACTIVATION = act.ReLU
LOSS_FN = loss.CrossEntropyLoss
LAYERS = (10,)
EPOCHS = 1000
ITER = 18

DATASET = load_iris()
INPUTS = DATASET.data
OUTPUTS = DATASET.target
X_train, X_test, Y_train, Y_test = train_test_split(INPUTS, OUTPUTS, test_size=0.2, random_state=42)

gr_descent = nn.Net(hidden_layers=LAYERS, activation=ACTIVATION, loss_function=LOSS_FN, lr=LEARNING_RATE,
                    save_metrics=True)
stgr_descent = nn.Net(hidden_layers=LAYERS, activation=ACTIVATION, loss_function=LOSS_FN, lr=LEARNING_RATE,
                      descent="stochastic", save_metrics=True)
gr_descent_dropout = nn.Net(hidden_layers=LAYERS, activation=ACTIVATION, loss_function=LOSS_FN, lr=LEARNING_RATE,
                            save_metrics=True, dropout=True)
stgr_descent_dropout = nn.Net(hidden_layers=LAYERS, activation=ACTIVATION, loss_function=LOSS_FN, lr=LEARNING_RATE,
                              descent="stochastic",
                              save_metrics=True, dropout=True)

gr_descent.fit(X_train, Y_train, epochs=EPOCHS)
stgr_descent.fit(X_train, Y_train, iter=ITER, epochs=EPOCHS)
gr_descent_dropout.fit(X_train, Y_train, epochs=EPOCHS)
stgr_descent_dropout.fit(X_train, Y_train, iter=ITER, epochs=EPOCHS)
result = gr_descent.predict(X_test)
result_st = stgr_descent.predict(X_test)
result_drop = gr_descent_dropout.predict(X_test)
result_st_drop = stgr_descent_dropout.predict(X_test)
# print(Y_test)
# print(result)
print("Accuracy: {}".format(np.sum(result == Y_test) / len(Y_test)))
print("Accuracy stochastic: {}".format(np.sum(result_st == Y_test) / len(Y_test)))
print("Accuracy with dropout: {}".format(np.sum(result_drop == Y_test) / len(Y_test)))
print("Accuracy stochastic with dropout: {}".format(np.sum(result_st_drop == Y_test) / len(Y_test)))

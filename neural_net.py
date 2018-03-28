import numpy as np

# Fix random seed for debug reasons
np.random.seed(1)


class Net:
    """
    Neural Net Class. Used to generate and train a model.
    """

    def __init__(self, inputs, outputs, layers=(), lr=0.01):
        """
        Initialisation of our neural network
        :param inputs: Input vectors
        :param outputs: Output vectors
        :param layers: Number of nodes per layer (only linear layers are supported)
        :param lr: Learning rate
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.input_shape = self.inputs.shape[1]
        self.output_shape = self.outputs.shape[1]
        self.layers = layers
        self.weights = None
        self.learning_rate = lr
        self.model = None
        self._intialize_weights()

    def _intialize_weights(self):
        """
        Initialize weights randomly
        """
        self.weights = []

        w1 = np.random.randn(self.input_shape, self.layers[0])
        self.weights.append(w1)

        for i in range(len(self.layers) - 1):
            wi = np.random.randn(self.layers[i], self.layers[i + 1])
            self.weights.append(wi)

        wn = np.random.randn(self.layers[-1], self.output_shape)
        self.weights.append(wn)

    def calculate_loss(self, pred_outputs):
        """
        Compute loss function (Cross entropy loss to begin with)
        :param pred_outputs: Predicted output via neural network
        :return: Value of loss
        """
        return -1/self.outputs.shape[0] * np.sum(self.outputs*np.log(pred_outputs) + (1-self.outputs)*np.log(1-pred_outputs))

    def forward(self):
        """
        Execute forward pass (not sure if should be used)
        """
        layers_nodes = []
        pred = Sigmoid.f(self.inputs.dot(self.weights[0]))
        layers_nodes.append(pred)
        for i in range(1, len(self.weights) - 1):
            pred = Sigmoid.f(pred.dot(self.weights[i]))
            layers_nodes.append(pred)
        pred = Sigmoid.f(pred.dot(self.weights[-1]))
        return pred, layers_nodes

    def backward(self, pred_outputs, nodes):
        """
        Compute backward propagation and update weights
        """
        pred_error = (pred_outputs - self.outputs)
        # print("prediction error: {}".format(pred_error))
        delta = pred_error * Sigmoid.derivative(pred_outputs)
        # print("delta: {}".format(delta))
        weight_adjustment = nodes[-1].T.dot(delta)
        # print("weight adjustment: {}".format(weight_adjustment))
        self.weights[-1] += weight_adjustment * self.learning_rate

        n = len(nodes)
        for i in range(1, n - 1):
            delta = delta.dot(self.weights[n - i].T) * Sigmoid.derivative(nodes[n - i])
            weight_adjustment = nodes[n - i].T.dot(delta)
            self.weights[n - i - 1] += weight_adjustment * self.learning_rate

        delta = delta.dot(self.weights[1].T) * Sigmoid.derivative(nodes[0])
        weight_adjustment = self.inputs.T.dot(delta)
        self.weights[0] += weight_adjustment * self.learning_rate
        return

    def train(self, n_iter=500):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Repeat
        """
        for t in range(n_iter):
            pred, nodes = self.forward()

            loss = self.calculate_loss(pred)
            print(t, loss)

            self.backward(pred, nodes)

    def predict(self):
        """
        Predict with the model (equivalent to a forward pass)
        """
        pred = Sigmoid.f(self.inputs.dot(self.weights[0]))
        for i in range(1, len(self.weights)):
            pred = Sigmoid.f(pred.dot(self.weights[i]))
        return pred


class Activation:
    """Interface regrouping methods of activation function"""

    @staticmethod
    def f(x):
        raise NotImplemented()

    @staticmethod
    def derivative(x):
        raise NotImplemented()


class Sigmoid(Activation):
    """
    Sigmoid function.
    Ref: https://en.wikipedia.org/wiki/Sigmoid_function
    """

    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))


class Softmax(Activation):
    """
    Softmax function.
    Ref: https://en.wikipedia.org/wiki/Softmax_function
    """

    @staticmethod
    def f(X):
        shift_X = np.exp(X - np.max(X))
        return shift_X / np.sum(shift_X)

    @staticmethod
    def derivative(x):
        raise NotImplemented()

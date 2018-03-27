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
        raise NotImplemented()

    def forward(self):
        """
        Execute forward pass (not sure if should be used)
        """
        raise NotImplemented()

    def backward(self):
        """
        Compute backward propagation and update weights
        """
        raise NotImplemented()

    def train(self):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Repeat
        """
        y_pred = self.weights[0].dot(self.weights[1])
        for i in range(2, len(self.weights)):
            y_pred = y_pred.dot(self.weights[i])
        return y_pred

    def predict(self):
        """
        Predict with the model (equivalent to a forward pass)
        """
        raise NotImplemented()


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

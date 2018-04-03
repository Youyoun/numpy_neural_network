import numpy as np

# Fix random seed for debug reasons
np.random.seed(1)


class Net:
    """
    Neural Net Class. Used to generate and train a model.
    """

    def __init__(self, inputs, outputs, layers=(), lr=0.01, random=True):
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
        self._intialize_weights(random)

    def _intialize_weights(self, random=True):
        """
        Initialize weights randomly
        """
        self.weights = []

        if random:
            w1 = np.random.randn(self.input_shape, self.layers[0])
            self.weights.append(w1)

            for i in range(len(self.layers) - 1):
                wi = np.random.randn(self.layers[i], self.layers[i + 1])
                self.weights.append(wi)

            wn = np.random.randn(self.layers[-1], self.output_shape)
            self.weights.append(wn)

        else:
            FIX_WEIGHT = 0.5
            w1 = np.full((self.input_shape, self.layers[0]), FIX_WEIGHT)
            self.weights.append(w1)

            for i in range(len(self.layers) - 1):
                wi = np.full((self.layers[i], self.layers[i + 1]), FIX_WEIGHT)
                self.weights.append(wi)

            wn = np.full((self.layers[-1], self.output_shape), FIX_WEIGHT)
            self.weights.append(wn)

    def cross_entropy_loss(self, pred_outputs):
        """
        Compute loss function (Cross entropy loss)
        Formula: E = -1/n * Sum_to_n(yi * log(yi) + (1-yi)*log(1-yi))
        :param pred_outputs: Predicted output via neural network
        :return: Value of loss
        """
        return -1/self.outputs.shape[0] * np.sum(self.outputs*np.log(pred_outputs) + (1-self.outputs)*np.log(1-pred_outputs))

    def mean_squared_error(self, pred_outputs):
        """
        Compute loss function (Mean squared error)
        Formula: E= 1/2 * (target - out)^2
        :param pred_outputs: Predicted output via neural network
        :return: value of loss
        """
        return 1/2 * np.sum(np.square(self.outputs - pred_outputs))

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
        delta = pred_error * Sigmoid.derivative(pred_outputs)
        delta = nodes[-1].T.dot(delta)
        self.weights[-1] -= delta * self.learning_rate

        n = len(nodes)
        for i in range(1, n - 1):
            delta = delta.T.dot(self.weights[n - i]) * Sigmoid.derivative(nodes[n - i])
            weight_adjustment = nodes[n - i].T.dot(delta)
            self.weights[n - i - 1] -= weight_adjustment * self.learning_rate

        delta = delta.dot(self.weights[1].T) * Sigmoid.derivative(nodes[0])
        weight_adjustment = self.inputs.T.dot(delta)
        self.weights[0] -= weight_adjustment * self.learning_rate
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

            loss = self.cross_entropy_loss(pred)
            print(t, loss)

            self.backward(pred, nodes)

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        """
        pred = Sigmoid.f(input.dot(self.weights[0]))
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

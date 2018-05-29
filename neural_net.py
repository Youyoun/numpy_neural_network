import numpy as np


# Fix random seed for debug reasons
# np.random.seed(1)


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
    def f(x):
        shift_x = np.exp(x - np.max(x))
        return shift_x / np.sum(shift_x)

    @staticmethod
    def derivative(x):
        return Softmax.f(x) * (1 - Softmax.f(x))


class ReLU(Activation):
    """
    RELU activation function
    Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """

    @staticmethod
    def f(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x):
        return (x > 0).astype(int)


class Softplus(Activation):
    """
    Softplus activation function
    Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """

    @staticmethod
    def f(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def derivative(x):
        return Sigmoid.f(x)


class Tanh(Activation):
    """
    Softplus activation function
    Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.square(np.tanh(x))


class Loss:
    @staticmethod
    def f(y, y_pred):
        raise NotImplemented()

    @staticmethod
    def delta(y, y_pred):
        raise NotImplemented()


class CrossEntropyLoss(Loss):
    @staticmethod
    def f(y, y_pred):
        """
        Compute loss function (Cross entropy loss)
        Formula: E = -1/n * Sum_to_n(yi * log(yi) + (1-yi)*log(1-yi))
        :param pred_outputs: Predicted output via neural network
        :return: Value of loss
        """
        return -1 / y.shape[1] * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def delta(y, y_pred):
        return 1 / len(y_pred) * (y_pred - y)


class MeanSquaredError(Loss):
    @staticmethod
    def f(y, y_pred):
        """
        Compute loss function (Mean squared error)
        Formula: E= 1/2 * (target - out)^2
        :param pred_outputs: Predicted output via neural network
        :return: value of loss
        """
        return 1 / (2 * len(y_pred)) * np.sum(np.square(y - y_pred))

    @staticmethod
    def delta(y, y_pred):
        return 1 / len(y_pred) * (y_pred - y)


class Net:
    """
    Neural Net Class. Used to generate and train a model.
    """

    def __init__(self, layers=(), activation=Sigmoid, loss=CrossEntropyLoss, lr=0.01, random=True):
        """
        Initialisation of our neural network
        :param inputs: Input vectors
        :param outputs: Output vectors
        :param layers: Number of nodes per layer (only linear layers are supported)
        :param lr: Learning rate
        """
        self.layers = layers
        self.weights = None
        self.biases = None
        self.learning_rate = lr
        self.loss = loss
        self._intialize_weights(random)
        self.activation = activation

    def _intialize_weights(self, random=True):
        """
        Initialize weights randomly
        """
        weights = []
        biases = []
        if random:
            for i in range(len(self.layers) - 1):
                wi = np.random.uniform(-1, 1, (self.layers[i + 1], self.layers[i]))
                weights.append(wi)
                biases.append(np.random.uniform(1, 0, (self.layers[i + 1], 1)))
        else:
            fix_weight = 0.5
            for i in range(len(self.layers) - 1):
                wi = np.full((self.layers[i + 1], self.layers[i]), fix_weight)
                weights.append(wi)
                biases.append(np.full((self.layers[i + 1], 1), fix_weight))
        self.weights, self.biases = np.array(weights), np.array(biases)

    def check_dimensions(self, inputs, outputs):
        if self.layers[0] != inputs.shape[0]:
            raise ValueError(
                "Incorrect dimension of features: Expected {} got {}".format(self.layers[0], inputs.shape[0]))
        elif self.layers[-1] != outputs.shape[0]:
            raise ValueError(
                "Incorrect dimension of outputs: Expected {} got {}".format(self.layers[-1], outputs.shape[0]))
        return

    def adjust_weights(self, nabla_weights, nabla_bias):
        for i in range(len(self.layers) - 1):
            self.weights[i] -= self.learning_rate * nabla_weights[i]
            if self.biases[i].shape == (1,):
                self.biases[i] -= self.learning_rate * np.array([[np.mean(e)] for e in nabla_bias[i]])[0]
            else:
                self.biases[i] -= self.learning_rate * np.array([[np.mean(e)] for e in nabla_bias[i]])
        return

    def backward(self, inputs, outputs, pred_outputs, nodes):
        """
        Compute backward propagation and update weights
        """
        adjustments = []
        deltas = []

        # Last layer
        pred_error = self.loss.delta(outputs, pred_outputs)
        delta = pred_error * self.activation.derivative(nodes[-1])
        weight_adjustment = np.dot(delta, nodes[-2].T)
        adjustments.insert(0, weight_adjustment)
        deltas.insert(0, delta)

        # Mid layers
        for i in reversed(range(2, len(nodes))):
            delta = np.dot(self.weights[i].T, delta) * self.activation.derivative(nodes[i - 1])
            weight_adjustment = np.dot(delta, nodes[i - 1].T)
            adjustments.insert(0, weight_adjustment)
            deltas.insert(0, delta)

        # First layers
        delta = np.dot(self.weights[1].T, delta) * self.activation.derivative(nodes[0])
        weight_adjustment = np.dot(delta, inputs.T)
        adjustments.insert(0, weight_adjustment)
        deltas.insert(0, delta)

        return adjustments, deltas

    def forward(self, inputs):
        """
        Execute forward pass (not sure if should be split from back propagation)
        """
        layers_nodes = []

        # First layer
        net_value = np.dot(self.weights[0], inputs) + self.biases[0]
        out_value = self.activation.f(net_value)
        layers_nodes.append(net_value)

        # Mid and Last layers
        for i in range(1, len(self.weights)):
            net_value = np.dot(self.weights[i], out_value) + self.biases[i]
            out_value = self.activation.f(net_value)
            layers_nodes.append(net_value)
        return out_value, layers_nodes

    def fit(self, inputs, outputs, n_iter=500, epochs=10):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Adjust weights
        - Repeat
        """
        self.check_dimensions(inputs, outputs)
        for t in range(epochs):
            for i in range(n_iter):
                pred, nodes = self.forward(inputs)
                loss = self.loss.f(outputs, pred)
                weight_adjustment, bias_adjustments = self.backward(inputs, outputs, pred, nodes)
                self.adjust_weights(weight_adjustment, bias_adjustments)
            print("Epoch: {} Loss: {} Mean difference: {}".format(t, loss, np.mean(outputs - pred)))

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        """
        input = np.array(input)
        pred = self.activation.f(np.dot(self.weights[0], input))
        for i in range(1, len(self.weights) - 1):
            pred = self.activation.f(np.dot(self.weights[i], pred))
        pred = self.activation.f(np.dot(self.weights[-1], pred))
        return pred

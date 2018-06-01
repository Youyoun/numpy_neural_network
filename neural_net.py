import numpy as np


# Fix random seed for debug reasons
np.random.seed(1)


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
        return -1 / len(y) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def delta(y, y_pred):
        return 1 / len(y_pred) * (y_pred - y)


# Utils
def shuffle(x, y):
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    return x[randomize, :], y[randomize, :]


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
        if random:
            self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
            self.weights = [np.random.randn(x, y) / np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        else:
            weights, biases = [], []
            fix_weight = 0.5
            for i in range(len(self.layers) - 1):
                wi = np.full((self.layers[i + 1], self.layers[i]), fix_weight)
                weights.append(wi)
                biases.append(np.full((self.layers[i + 1], 1), fix_weight))
            self.weights, self.biases = np.array(weights), np.array(biases)

    def check_dimensions(self, inputs, outputs):
        if self.layers[0] != inputs.shape[1]:
            raise ValueError(
                "Incorrect dimension of features: Expected {} got {}".format(self.layers[0], inputs.shape[0]))
        try:
            if self.layers[-1] != outputs.shape[1]:
                raise ValueError(
                    "Incorrect dimension of outputs: Expected {} got {}".format(self.layers[-1], outputs.shape[0]))
        except IndexError:
            if self.layers[-1] != 1:
                raise ValueError(
                    "Incorrect dimension of outputs: Expected {} got {}".format(self.layers[-1], 1))
        return

    def adjust_weights(self, nabla_weights, nabla_bias):
        for i in range(len(self.layers) - 1):
            self.weights[i] -= self.learning_rate * nabla_weights[i]
            for j in range(len(nabla_bias)):
                if self.biases[i].shape == (1,):
                    self.biases[i] -= self.learning_rate * nabla_bias[i][j][0]
                else:
                    self.biases[i] -= self.learning_rate * nabla_bias[i][j]
        return

    def back_propagation(self, inputs, outputs):

        bias_adjustments = []
        weight_adjustments = []

        # forward pass
        activation = inputs
        activations = [activation]
        layers_nodes = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            layers_nodes.append(z)
            activation = self.activation.f(z)
            activations.append(activation)

        # backpass
        delta = self.loss.delta(activations[-1], outputs)
        bias_adjustments.append(delta)
        nabla_w = np.dot(activations[-2].T, delta)
        weight_adjustments.append(nabla_w)
        for i in range(2, len(self.layers)):
            delta = np.dot(delta, self.weights[-i + 1].T) * self.activation.derivative(layers_nodes[- i])
            bias_adjustments.insert(0, delta)
            nabla_w = np.dot(activations[-i - 1].T, delta)
            weight_adjustments.insert(0, nabla_w)
        return weight_adjustments, bias_adjustments

    def fit(self, inputs, outputs, n_iter=500):
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
        for i in range(n_iter):
            weight_adjustment, bias_adjustments = self.back_propagation(inputs, outputs)
            self.adjust_weights(weight_adjustment, bias_adjustments)
            pred = self.predict(inputs)
            loss = self.loss.f(outputs, pred)
            print("Iteration: {} Loss: {}".format(i, loss, np.mean(outputs - pred)))

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        """
        input = np.array(input)
        pred = self.activation.f(np.dot(input, self.weights[0]) +  self.biases[0])
        for i in range(1, len(self.weights)):
            pred = self.activation.f(np.dot(pred, self.weights[i]) + self.biases[i])
        return pred


class StochasticNet(Net):
    def __init__(self, layers=(), activation=Sigmoid, loss=CrossEntropyLoss, lr=0.01, random=True):
        super().__init__(layers=layers, activation=activation, loss=loss, lr=lr, random=random)

    def fit(self, inputs, outputs, batch_num=2, epochs=1000):
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
            training_set = shuffle(inputs, outputs)
            x_batches = np.array_split(training_set[0], batch_num, axis=0)
            y_batches = np.array_split(training_set[1], batch_num, axis=0)
            for x, y in zip(x_batches, y_batches):
                weight_adjustment, bias_adjustments = self.back_propagation(x, y)
                self.adjust_weights(weight_adjustment, bias_adjustments)
            pred = self.predict(inputs)
            loss = self.loss.f(outputs, pred)
            print("Epoch: {} Loss: {} Mean difference: {}".format(t, loss, np.mean(outputs - pred)))

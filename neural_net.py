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


class Net:
    """
    Neural Net Class. Used to generate and train a model.
    """

    def __init__(self, layers=(), mid_layer_activation=Sigmoid, output_layer_activation=Sigmoid, lr=0.01, random=True):
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
        self.model = None
        self._intialize_weights(random)

        self.activation = mid_layer_activation
        self.output_activation = output_layer_activation

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

    def cross_entropy_loss(self, real_outputs, pred_outputs):
        """
        Compute loss function (Cross entropy loss)
        Formula: E = -1/n * Sum_to_n(yi * log(yi) + (1-yi)*log(1-yi))
        :param pred_outputs: Predicted output via neural network
        :return: Value of loss
        """
        return -1 / real_outputs.shape[1] * np.sum(
            real_outputs * np.log(pred_outputs) + (1 - real_outputs) * np.log(1 - pred_outputs))

    def mean_squared_error(self, real_outputs, pred_outputs):
        """
        Compute loss function (Mean squared error)
        Formula: E= 1/2 * (target - out)^2
        :param pred_outputs: Predicted output via neural network
        :return: value of loss
        """
        return 1 / (2 * len(pred_outputs)) * np.sum(np.square(real_outputs - pred_outputs))

    def forward(self, inputs):
        """
        Execute forward pass (not sure if should be used)
        """
        layers_nodes = []

        # First layer
        net_value = np.dot(self.weights[0], inputs) + self.biases[0]
        out_value = self.activation.f(net_value)
        layers_nodes.append(net_value)

        # Mid layers
        for i in range(1, len(self.weights) - 1):
            net_value = np.dot(self.weights[i], out_value) + self.biases[i]
            out_value = self.activation.f(net_value)
            layers_nodes.append(net_value)

        # Last layer
        net_value = np.dot(self.weights[-1], out_value) + self.biases[-1]
        out_value = self.output_activation.f(net_value)
        layers_nodes.append(net_value)
        return out_value, layers_nodes

    def backward(self, inputs, outputs, pred_outputs, nodes):
        """
        Compute backward propagation and update weights
        """
        adjustments = []
        deltas = []

        # Last layer
        pred_error = - (1 / len(pred_outputs)) * (outputs - pred_outputs)
        delta = pred_error * self.output_activation.derivative(nodes[-1])
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

        # Update weights
        for i in range(len(adjustments)):
            self.weights[i] -= self.learning_rate * adjustments[i]
            if self.biases[i].shape == (1,):
                self.biases[i] -= self.learning_rate * np.array([[np.mean(e)] for e in deltas[i]])[0]
            else:
                self.biases[i] -= self.learning_rate * np.array([[np.mean(e)] for e in deltas[i]])

    def fit(self, inputs, outputs, n_iter=500, epochs=10):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Repeat
        """
        for t in range(epochs):
            for i in range(n_iter):
                pred, nodes = self.forward(inputs)
                loss = self.mean_squared_error(outputs, pred)
                self.backward(inputs, outputs, pred, nodes)
            print("Epoch: {} Loss: {} Mean difference: {}".format(t, loss, np.mean(outputs - pred)))

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        """
        input = np.array(input)
        pred = self.activation.f(np.dot(self.weights[0], input))
        for i in range(1, len(self.weights) - 1):
            pred = self.activation.f(np.dot(self.weights[i], pred))
        pred = self.output_activation.f(np.dot(self.weights[-1], pred))
        return pred

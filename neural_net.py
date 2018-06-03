import numpy as np
import activation as act
import loss
import utils

# Fix random seed for debug reasons
np.random.seed(0)

class Net:
    """
    Neural Net Class. Used to generate and train a model.
    """

    def __init__(self, hidden_layers=(), activation=act.ReLU, loss_function=loss.CrossEntropyLoss, lr=0.01, descent=None):
        """
        Initialisation of our neural network
        :param inputs: Input vectors
        :param outputs: Output vectors
        :param layers: Number of nodes per layer (only linear layers are supported)
        :param lr: Learning rate
        """
        self.weights = None
        self.biases = None
        self.learning_rate = lr
        self.loss = loss_function
        if self.loss == loss.CrossEntropyLoss:
            self.is_classifier = True
            self.outer_activation = act.Softmax
        else:
            self.is_classifier = False
            self.outer_activation = act.Identity
        self.layers = list(hidden_layers)
        self.n_layers = len(self.layers) + 1
        self.descent = descent
        self.activation = activation
        self.first_fit = True

    def _set_input_output_layer(self, x, y):
        input_shape = x.shape[1]
        try:
            output_shape = y.shape[1]
        except IndexError:
            output_shape = 1

        self.layers.insert(0, input_shape)
        if self.is_classifier:
            self.labels = self._get_classes(y)
            self.n_classes = len(self.labels)
            self.layers.append(self.n_classes)
        else:
            self.layers.append(output_shape)
        print(self.layers, self.labels)
        return

    def _get_classes(self, y):
        labels = np.unique(y)
        return {i:labels[i] for i in range(len(labels))}

    def _fit_labels(self, y):
        new_labels = [[0]*self.n_classes for i in range(len(y))]
        for i in range(len(y)):
            new_labels[i][self.labels[y[i]]] = 1
        return np.array(new_labels)

    def _reverse_labels(self, y_pred):
        return np.array([self.labels[np.argmax(e)] for e in y_pred])

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

    def feed_forward(self, input):
        pred = np.array(input)
        for i in range(self.n_layers):
            if not i == self.n_layers - 1:
                pred = self.activation.f(np.dot(pred, self.weights[i]) + self.biases[i])
            else:
                pred = self.outer_activation.f(np.dot(pred, self.weights[i]) + self.biases[i])
        return pred

    def back_propagation(self, inputs, outputs):

        bias_adjustments = []
        weight_adjustments = []

        # forward pass
        activation = inputs
        activations = [activation]
        layers_nodes = []
        for i in range(self.n_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            layers_nodes.append(z)
            if i == self.n_layers - 1: #last layer
                activation = self.outer_activation.f(z)
            else:
                activation = self.activation.f(z)
            activations.append(activation)

        # backpass
        delta = self.loss.delta(outputs, activations[-1])
        bias_adjustments.append(np.mean(delta, 0))
        nabla_w = np.dot(activations[-2].T, delta)
        weight_adjustments.append(nabla_w)
        for i in range(2, len(self.layers)):
            delta = np.dot(delta, self.weights[-i + 1].T) * self.activation.derivative(layers_nodes[- i])
            bias_adjustments.insert(0, np.mean(delta, 0))
            nabla_w = np.dot(activations[-i - 1].T, delta)
            weight_adjustments.insert(0, nabla_w)
        return weight_adjustments, bias_adjustments

    def _fit(self, inputs, outputs, epochs=500):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Adjust weights
        - Repeat
        """
        for i in range(epochs):
            weight_adjustment, bias_adjustments = self.back_propagation(inputs, outputs)
            self.adjust_weights(weight_adjustment, bias_adjustments)
            pred = self.feed_forward(inputs)
            loss = self.loss.f(outputs, pred)
            if self.is_classifier:
                acc = np.sum(self._reverse_labels(outputs) == self._reverse_labels(pred)) / len(outputs) * 100
                print("Iteration: {} Loss: {} Accuracy: {}".format(i, loss, acc))
            else:
                print("Iteration: {} Loss: {}".format(i, loss))

    def _fit_stochastic(self, inputs, outputs, iter=2, epochs=1000):
        """
        Train the model.
        Steps:
        - Forward Pass
        - Compute Loss
        - Backward Pass
        - Adjust weights
        - Repeat
        """
        for t in range(epochs):
            training_set = utils.shuffle(inputs, outputs)
            x_batches = np.array_split(training_set[0], iter, axis=0)
            y_batches = np.array_split(training_set[1], iter, axis=0)
            for x, y in zip(x_batches, y_batches):
                weight_adjustment, bias_adjustments = self.back_propagation(x, y)
                self.adjust_weights(weight_adjustment, bias_adjustments)
            pred = self.feed_forward(inputs)
            loss = self.loss.f(outputs, pred)
            if self.is_classifier:
                acc = np.sum(self._reverse_labels(outputs) == self._reverse_labels(pred)) / len(outputs) * 100
                print("Epoch: {} Loss: {} Accuracy: {}".format(t, loss, acc))
            else:
                print("Epoch: {} Loss: {}".format(t, loss))

    def fit(self, inputs, outputs, **kwargs):
        if self.first_fit:
            self._set_input_output_layer(inputs, outputs)
            self._intialize_weights()
            self.first_fit = False
        if self.is_classifier:
            labels = self._fit_labels(outputs)
        else:
            labels = outputs.copy()
        if self.descent == 'stochastic':
            return self._fit_stochastic(inputs, labels, **kwargs)
        elif self.descent is None:
            return self._fit(inputs, labels, **kwargs)
        else:
            raise ValueError("Incorrect descent method (only accept Stochastic or None)")

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        """
        pred = self.feed_forward(input)
        if not self.is_classifier:
            return pred
        else:
            return self._reverse_labels(pred)


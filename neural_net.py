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

    def __init__(self, hidden_layers=(), activation=act.ReLU, loss_function=loss.CrossEntropyLoss, lr=0.01,
                 descent=None, save_metrics=False, dropout=False):
        """
        Initialisation of our neural network

        :param hidden_layers: tuple, length = number of hidden layers and int = number of nodes in layer Ex:(10,5)
            represent 2 hidden layers, first with 10 nodes and second with 5
        :param activation: activation function on hidden layers
        :param loss_function: loss function. if CrossEntropy, classification is used.
        :param lr: learning rate
        :param descent: if set to "stochastic", use stochastic gradient descent
        :param save_metrics: if True, save some metrics for graphs
        :param dropout: if True, enable Srivasta drop out
        """
        self.weights = None
        self.biases = None
        self.learning_rate = lr
        self.loss = loss_function

        # If cross entropy, Net is a classifier
        if self.loss == loss.CrossEntropyLoss:
            self.is_classifier = True
            self.outer_activation = act.Softmax
        else:
            self.is_classifier = False
            self.outer_activation = act.Identity

        # Set a few additionnal parameters for our net
        self.layers = list(hidden_layers)  # List of hidden layers, will be updated on first fit
        self.n_layers = len(self.layers) + 1  # Total number of layers
        self.descent = descent
        self.activation = activation

        self.first_fit = True  # If true, net has never been trained

        if save_metrics:
            self.metrics = {
                "epoch": [],
                "iter": [],
                "loss": [],
                "accuracy": [],
                "lr": [],
            }
        else:
            self.metrics = None

        # Dropout parmeters
        self.srivastava = dropout
        self.entry_proba = 0.5
        self.hidden_proba = 0.5

    def _set_input_output_layer(self, x, y):
        """
        Add layers for input and define few parameters
        :param x: features vector
        :param y: labels vector
        """
        input_shape = x.shape[1]
        try:  # Sometimes, array is in shape (n_sample,) This means that output value is only one
            output_shape = y.shape[1]
        except IndexError:
            output_shape = 1

        self.layers.insert(0, input_shape)  # add input shape

        if self.is_classifier:  # If net is classifier, define labels and classes and add last layer
            self.labels = self._get_classes(y)
            self.n_classes = len(self.labels)
            self.layers.append(self.n_classes)  # Last layer contains nodes number = to number of classes
        else:
            self.layers.append(output_shape)
        return

    def _get_classes(self, y):
        """
        Get all classes from labels vector
        :param y: Labels vector
        :return: dictionary of classes indexed from 1 to n_classes
        """
        labels = np.unique(y)
        return {i: labels[i] for i in range(len(labels))}

    def _fit_labels(self, y):
        """
        In case of classification, transform label vector to vector of binary values
        For example: if classes are 0,1,2, then fit_labels returns vector [1,0,0], [0,1,0], [0,0,1] for classes
        respectively 0,1,2
        :param y: labels vector
        :return: array of new labels
        """
        new_labels = [[0] * self.n_classes for i in range(len(y))]
        for i in range(len(y)):
            new_labels[i][self.labels[y[i]]] = 1
        return np.array(new_labels)

    def _reverse_labels(self, y_pred):
        """
        Reverse fit_labels method, to obtain the class of the output vectors from the neural net, in case of classification
        :param y_pred: vector of binary labels
        :return: array of labels vector
        """
        return np.array([self.labels[np.argmax(e)] for e in y_pred])

    def _intialize_weights(self):
        """
        Initialize weights randomly
        """
        self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]

    def check_dimensions(self, inputs, outputs):
        """
        Check dimensions of inputs and outputs sothat they correspond to the layers
        :param inputs: Features Vector
        :param outputs: Labels Vector
        """
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
        """
        Adjust weight of the Net using their gradients
        :param nabla_weights: gradient of weight
        :param nabla_bias: gradient of biases
        :return:
        """
        for i in range(len(self.layers) - 1):
            self.weights[i] -= self.learning_rate * nabla_weights[i]
            for j in range(len(nabla_bias[i])):
                if self.biases[i].shape == (1,):
                    self.biases[i] -= self.learning_rate * nabla_bias[i][j][0]
                else:
                    self.biases[i] -= self.learning_rate * nabla_bias[i][j]
        return

    def feed_forward(self, input):
        """
        Execute a forward pass.
        :param input: Features vectors
        :return: Predicted Labels
        """
        pred = np.array(input)
        for i in range(self.n_layers):
            if not i == self.n_layers - 1:
                pred = self.activation.f(np.dot(pred, self.weights[i]) + self.biases[i])
            else:
                pred = self.outer_activation.f(np.dot(pred, self.weights[i]) + self.biases[i])
        return pred

    def back_propagation(self, inputs, outputs):
        """
        Execute a full back propagation algorithm
        Steps:
        - Forward pass
        - Backward pass (compute gradients)
        - Adjust weights
        If dropout is activated, additionnal step where we update prior weights using dropout ones is added
        :param inputs: Feature vectors
        :param outputs: Labels Vectors
        """

        # Lists contaning gradients
        bias_adjustments = []
        weight_adjustments = []

        # forward pass
        activation = inputs

        if self.srivastava:  # If dropout is activated, back up weights and biases, and remove input nodes and weight accordingly
            self.backup_weights = self.weights.copy()
            self.backup_biases = self.biases.copy()
            masks = []  # Masks contains arrays that dictate what colums and rows have been removed (0 for removed and 1 for kept)
            activation, self.weights[0], _, mask = self.dropout(0, activation, proba=self.entry_proba)
            masks.append(mask)

        activations = [activation]
        layers_nodes = []
        for i in range(self.n_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]

            # If dropout is activaed and it is not the last layer
            if self.srivastava and not i == self.n_layers - 1:
                z, self.weights[i + 1], self.weights[i], mask = self.dropout(i + 1, z, proba=self.hidden_proba)
                masks.append(mask)

            layers_nodes.append(z)
            if i == self.n_layers - 1:  # last layer
                activation = self.outer_activation.f(z)
            else:
                activation = self.activation.f(z)
            activations.append(activation)

        # backward pass

        # Last layer
        delta = self.loss.delta(outputs, activations[-1])
        bias_adjustments.append(np.mean(delta, 0))
        nabla_w = np.dot(activations[-2].T, delta)
        weight_adjustments.append(nabla_w)
        # Hidden layers
        for i in range(2, len(self.layers)):
            delta = np.dot(delta, self.weights[-i + 1].T) * self.activation.derivative(layers_nodes[- i])
            bias_adjustments.insert(0, np.mean(delta, 0))
            nabla_w = np.dot(activations[-i - 1].T, delta)
            weight_adjustments.insert(0, nabla_w)

        # Adjust weights with gradients
        self.adjust_weights(weight_adjustments, bias_adjustments)

        # If dropout, update weights into old weights
        if self.srivastava:
            # Hidden layers
            for i in range(self.n_layers - 1):
                self.weights[i] = self.fix_dropout(self.weights[i], self.backup_weights[i], masks[i], masks[i + 1])
            # Last Layer
            self.weights[-1] = self.fix_dropout(self.weights[-1], self.backup_weights[-1], masks[-1], None)

        return

    def _fit(self, inputs, outputs, epochs=500):
        for i in range(epochs):
            self.back_propagation(inputs.copy(), outputs)
            pred = self.feed_forward(inputs)
            loss = self.loss.f(outputs, pred)
            if self.is_classifier:
                acc = np.sum(self._reverse_labels(outputs) == self._reverse_labels(pred)) / len(outputs) * 100
                print("Iteration: {} Loss: {} Accuracy: {}".format(i, loss, acc))
            else:
                print("Iteration: {} Loss: {}".format(i, loss))
            if self.metrics is not None:
                self.metrics["epoch"].append(i)
                if self.is_classifier:
                    self.metrics["accuracy"].append(acc)
                self.metrics["loss"].append(loss)
                self.metrics["lr"].append(self.learning_rate)

    def _fit_stochastic(self, inputs, outputs, iter=2, epochs=1000):
        for t in range(epochs):
            training_set = utils.shuffle(inputs.copy(), outputs)
            x_batches = np.array_split(training_set[0], iter, axis=0)
            y_batches = np.array_split(training_set[1], iter, axis=0)
            for x, y in zip(x_batches, y_batches):
                self.back_propagation(x, y)
            pred = self.feed_forward(inputs)
            loss = self.loss.f(outputs, pred)
            if self.is_classifier:
                acc = np.sum(self._reverse_labels(outputs) == self._reverse_labels(pred)) / len(outputs) * 100
                print("Epoch: {} Loss: {} Accuracy: {}".format(t, loss, acc))
            else:
                print("Epoch: {} Loss: {}".format(t, loss))
            if self.metrics is not None:
                self.metrics["epoch"].append(t)
                self.metrics["iter"].append(iter)
                if self.is_classifier:
                    self.metrics["accuracy"].append(acc)
                self.metrics["loss"].append(loss)
                self.metrics["lr"].append(self.learning_rate)

    def fit(self, inputs, outputs, **kwargs):
        """
        Training method for the neural net.
        If stochastic is on, use stochastic fit.
        Else use normal Fit.
        :param inputs: Features vector in the form of (n_samples, n_features)
        :param outputs: Label vectors. For classification, provide normal output vector contaning classes directly
        :param kwargs: Additionnal arguments for fit methods
        :return: None
        """
        if self.first_fit:  # Initialize weights, update layers
            self._set_input_output_layer(inputs, outputs)
            self._intialize_weights()
            self.first_fit = False

        if self.is_classifier:  # Adjust labels so that they become binary
            labels = self._fit_labels(outputs)
        else:
            labels = outputs

        # Exectute fit method
        if self.descent == 'stochastic':
            return self._fit_stochastic(inputs, labels, **kwargs)
        elif self.descent is None:
            return self._fit(inputs, labels, **kwargs)
        else:
            raise ValueError("Incorrect descent method (only accept Stochastic or None)")

    def dropout(self, layer_index, activation, proba=0.5, random_state=122):
        """
        Dropout method.
        Removes Nodes from activation, and rows or columns from weights.
        Removing nodes from ith activation corresponds to removing columns from activations[i]. To adjust weights,
        we remove line i from next weights and column i from previous weights.
        :param layer_index: Activation index
        :param activation: Nodes values of current layer
        :param proba: Probablity of removal
        :param random_state: State of rng (fix)
        :return: New activation, New next weights, New previous weights, Mask used for removals
        """
        rng = np.random.RandomState(random_state)
        mask = 1 - rng.binomial(size=(activation.shape[1],), n=1,
                                p=proba)  # Array of length features of activation nodes
        dropout_act = activation[:, mask == 1]
        next_weight_new = self.weights[layer_index][mask == 1, :]
        if layer_index == 0:  # No previous weight matrix for first layer
            prev_weight_new = None
        else:
            prev_weight_new = self.weights[layer_index - 1][:, mask == 1]
        return dropout_act, next_weight_new, prev_weight_new, mask

    def fix_dropout(self, new_weights, old_weights, mask_prev, mask_next=None):
        """
        Update old weight with new dropout weights.
        New weights don't have the same format as old ones, so updating them is pretty tricky.
        :param new_weights: Weights of the Net after dropout
        :param old_weights: Weights of the Net before dropout (Real weights)
        :param mask_prev: Mask used on previous nodes
        :param mask_next: Mask used on next nodes
        :return: New weights for the real Net
        """
        if mask_next is not None:
            new_w = old_weights
            if mask_next is not None:
                conf_matrix = utils.confusion_matrix(mask_next,
                                                     mask_prev)  # Compute confusion matrix containing indexes of weights to update
                indexes = np.argwhere(conf_matrix)
                c = 0
                for i in range(len(new_weights)):
                    for j in range(len(new_weights[i])):
                        new_w[indexes[c][0], indexes[c][1]] = new_weights[i, j]
                        c += 1
        else:
            new_w = self.fix_dropout(new_weights, old_weights, mask_prev, np.ones(old_weights.shape[1]))
        return new_w

    def predict(self, input):
        """
        Predict with the model (equivalent to a forward pass)
        :param input: Same format as fit method
        ;return: Predicted labels, real classes if classifier
        """
        if self.first_fit:
            raise ValueError("Train model before predicting")
        pred = self.feed_forward(input)
        if not self.is_classifier:
            return pred
        else:
            return self._reverse_labels(pred)

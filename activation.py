import numpy as np

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

class Identity(Activation):
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)
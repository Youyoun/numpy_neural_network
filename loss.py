import numpy as np

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

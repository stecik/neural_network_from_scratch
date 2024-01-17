import numpy as np


class MSE:
    def __call__(self, x, y):
        func = np.power(x - y, 2)
        deriv = 2 * (x - y)
        return (func, deriv)

    def __str__(self):
        return "MSE"

    def __repr__(self):
        return "MSE"


class CategoricalCrossentropy:
    def __init__(self, epsilon=1e-15) -> None:
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return (-np.sum(y_true * np.log(y_pred)) / y_true.shape[0], 1)

    def __str__(self):
        return "Categorical Crossentropy"

    def __repr__(self):
        return "Categorical Crossentropy"

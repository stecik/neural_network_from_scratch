import numpy as np


class ReLU:
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __call__(self, x):
        return np.maximum(0, x)

    def __str__(self):
        return "ReLU"

    def __repr__(self):
        return "ReLU"


class LeakyReLU:
    def __init__(self, alpha=0.1) -> None:
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)

    def __str__(self):
        return "LeakyReLU"

    def __repr__(self):
        return "LeakyReLU"


class ELU:
    def __init__(self, alpha=1) -> None:
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def __str__(self):
        return "ELU"

    def __repr__(self):
        return "ELU"


class SoftPlus:
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def __str__(self):
        return "SoftPlus"

    def __repr__(self):
        return "SoftPlus"


class Swish:
    def __call__(self, x):
        return x / (1 + np.exp(-x))

    def __str__(self):
        return "Swish"

    def __repr__(self):
        return "Swish"


class Softmax:
    def derivative(self, x):
        # not actual derivative, only works with CategoricalCrossentropy
        return 1

    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    def __str__(self):
        return "Softmax"

    def __repr__(self):
        return "Softmax"


class Sigmoid:
    def derivative(self, x):
        # Compute the Sigmoid of x
        sigmoid_x = self(x)
        # Compute the derivative using the formula for the derivative of the Sigmoid function
        return sigmoid_x * (1 - sigmoid_x)

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self):
        return "Sigmoid"

    def __repr__(self):
        return "Sigmoid"


class Tanh:
    def __call__(self, x):
        return np.tanh(x)

    def __str__(self):
        return "Tanh"

    def __repr__(self):
        return "Tanh"


class MSE:
    def derivative(self, x, y):
        return 2 * (x - y)

    def __call__(self, x, y):
        return np.power(x - y, 2)

    def __str__(self):
        return "MSE"

    def __repr__(self):
        return "MSE"


class CategoricalCrossentropy:
    def __init__(self, epsilon=1e-15) -> None:
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def __str__(self):
        return "Categorical Crossentropy"

    def __repr__(self):
        return "Categorical Crossentropy"

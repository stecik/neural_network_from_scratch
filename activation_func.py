import numpy as np


class ReLU:
    def __call__(self, x):
        func = np.maximum(0, x)
        deriv = np.where(x > 0, 1, 0)
        return (func, deriv)

    def __str__(self):
        return "ReLU"

    def __repr__(self):
        return "ReLU"


class LeakyReLU:
    def __init__(self, alpha=0.1) -> None:
        self.alpha = alpha

    def __call__(self, x):
        func = np.maximum(self.alpha * x, x)
        deriv = np.where(x > 0, 1, self.alpha)
        return (func, deriv)

    def __str__(self):
        return "LeakyReLU"

    def __repr__(self):
        return "LeakyReLU"


class ELU:
    def __init__(self, alpha=1) -> None:
        self.alpha = alpha

    def __call__(self, x):
        func = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        deriv = np.where(x > 0, 1, func + self.alpha)
        return (func, deriv)

    def __str__(self):
        return "ELU"

    def __repr__(self):
        return "ELU"


class SoftPlus:
    def __call__(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        func = np.log(sigmoid)
        deriv = 1 / (sigmoid)
        return (func, deriv)

    def __str__(self):
        return "SoftPlus"

    def __repr__(self):
        return "SoftPlus"


class Swish:
    def __call__(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        func = x * sigmoid
        deriv = sigmoid + func * (1 - sigmoid)
        return (func, deriv)

    def __str__(self):
        return "Swish"

    def __repr__(self):
        return "Swish"


class Softmax:
    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        func = exps / np.sum(exps, axis=0)
        return (func, 1)

    def __str__(self):
        return "Softmax"

    def __repr__(self):
        return "Softmax"


class Sigmoid:
    def __call__(self, x):
        func = 1 / (1 + np.exp(-x))
        deriv = func * (1 - func)
        return (func, deriv)

    def __str__(self):
        return "Sigmoid"

    def __repr__(self):
        return "Sigmoid"


class Tanh:
    def __call__(self, x):
        func = np.tanh(x)
        deriv = 1 - np.power(func, 2)
        return (func, deriv)

    def __str__(self):
        return "Tanh"

    def __repr__(self):
        return "Tanh"

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.utils import shuffle
import pickle

from tensorflow.keras.datasets import mnist
from functions import ReLU, Sigmoid, Softmax, MSE, CategoricalCrossentropy


class NeuralNetwork:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.layers = []
        self.loss = None
        self.best_model = []
        self.best_loss = np.inf

    def add(self, layer):
        self.layers.append(layer)

    def add_loss(self, loss):
        self.loss = loss

    def save_model(self, name):
        pickle.dump(self.best_model, open(name, "wb"))

    def load_model(self, name):
        self.layers = pickle.load(open(name, "rb"))

    def forward(self, input):
        for layer in self.layers:
            layer.input = input
            layer.forward()
            input = layer.a
        return input

    def forward_batch(self, input_batch, labels_batch):
        loss = 0
        for i, input in enumerate(input_batch):
            output = self.forward(input)
            loss += self.loss(output, labels_batch[i])
            # Check for softmax and categorical crossentropy
            if isinstance(self.loss, CategoricalCrossentropy) and isinstance(
                self.layers[-1].activation, Softmax
            ):
                err = self.layers[-1].a - labels_batch[i]
            else:
                err = self.loss.derivative(self.layers[-1].a, labels_batch[i])
            self.backward(err, input_batch.shape[0])
        return np.mean(loss)

    def backward(self, err, batch_size):
        for layer in reversed(self.layers):
            err = layer.backward(err, batch_size)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def train(self, epochs, batch_size, lr):
        print("Training...")
        for _ in range(epochs):
            loss = 0
            for i in range(len(self.input) // batch_size):
                input_batch = self.input[i * batch_size : (i + 1) * batch_size]
                labels_batch = self.labels[i * batch_size : (i + 1) * batch_size]
                loss = self.forward_batch(input_batch, labels_batch)
                self.update(lr)
            self.input, self.labels = shuffle(self.input, self.labels)
            print(f"Loss: {loss}")
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_model = self.layers.copy()

        print("Training finished")
        print(f"Best loss: {self.best_loss}")
        self.layers = self.best_model.copy()
        return self.layers

    def accuracy(self):
        pass

    def summary(self):
        for layer in self.layers:
            print(layer)
        print(f"Loss: {self.loss}")
        print()


class DenseLayer:
    def __init__(self, input_size, output_size, activation, use_bias=True):
        self.input = input
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = self.init_weights()
        self.biases = np.zeros(output_size)
        self.derivative = None
        self.z = None
        self.a = None
        self.weight_grad = 0
        self.use_bias = use_bias
        if use_bias:
            self.bias_grad = 0

    def init_weights(self):
        limit = np.sqrt(2 / self.input_size)
        return np.random.uniform(-limit, limit, (self.input_size, self.output_size))

    def forward(self):
        self.z = np.dot(self.input, self.weights) + self.biases
        self.a = self.activation(self.z)
        self.derivative = self.activation.derivative(self.z)
        self.input_reshaped = np.tile(self.input[:, None], (1, self.weights.shape[1]))

    def backward(self, err, batch_size):
        err = self.derivative * err
        if self.use_bias:
            self.bias_grad = err
        err = np.reshape(err, (1, self.weights.shape[1]))
        input = np.reshape(self.input, (self.weights.shape[0], 1))
        self.weight_grad = np.dot(input, err) / batch_size
        err = np.dot(err, self.weights.T)
        return err

    def update(self, lr):
        self.weights -= lr * self.weight_grad
        self.weight_grad = 0
        if self.use_bias:
            self.biases -= lr * np.reshape(self.bias_grad, self.biases.shape)
            self.bias_grad = 0

    def __str__(self) -> str:
        return f"Dense layer\nInput: {self.input_size}\nOutput: {self.output_size}\nActivation: {self.activation}\n---------------------------------"

    def __repr__(self) -> str:
        return f"Dense layer\nInput: {self.input_size}\nOutput: {self.output_size}\nActivation: {self.activation}\n---------------------------------"


def data_preprocessing():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    labels_enc = one_hot_enc(train_y)
    test_y_enc = one_hot_enc(test_y)
    input = train_x.reshape(60000, 784)
    test_x = test_x.reshape(10000, 784)
    return (input, labels_enc, test_x, test_y_enc)


def one_hot_enc(labels):
    unique_labels = set(labels)
    labels_enc = np.zeros((labels.shape[0], len(unique_labels)))
    for i in range(labels.shape[0]):
        labels_enc[i][labels[i]] = 1
    return labels_enc


def one_hot_dec(label):
    return np.argmax(label)


def accuracy(nn, x, y_enc):
    correct = 0
    for i in range(len(x)):
        pred = nn.forward(x[i])
        if one_hot_dec(pred) == one_hot_dec(y_enc[i]):
            correct += 1
    return correct / len(x) * 100


if __name__ == "__main__":
    start = time()

    # set data
    input, labels_enc, test_x, test_y_enc = data_preprocessing()

    # NN architecture
    nn = NeuralNetwork(input, labels_enc)
    nn.add(DenseLayer(784, 100, ReLU()))
    nn.add(DenseLayer(100, 50, ReLU()))
    nn.add(DenseLayer(50, 10, Softmax()))
    nn.add_loss(CategoricalCrossentropy())
    nn.summary()

    # hyperparameters (relu + softmax + categorical crossentropy)
    epochs = 30
    batch_size = 60
    lr = 0.0001

    # train
    # nn.train(epochs, batch_size, lr)
    end = time()
    print(f"Training time: {end - start}")
    # nn.save_model("models/model.pickle")
    nn.load_model("models/model2.pickle")

    print(f"Train Accuracy: {accuracy(nn, input, labels_enc)}%")
    print(f"Test Accuracy: {accuracy(nn, test_x, test_y_enc)}%")

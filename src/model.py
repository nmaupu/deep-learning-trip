import numpy as np
from sklearn.metrics import accuracy_score


class LearningModel:
    """A simple neuron learning model"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)

    def model(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        return A

    def predict(self, X):
        A = self.model(X)
        return A >= 0.5, A

    def log_loss(self, A):
        return (-1 / len(self.y)) * np.sum(
            self.y * np.log(A) + (1 - self.y) * np.log(1 - A)
        )

    def gradients(self, A):
        dW = (1 / len(self.y)) * np.dot(self.X.T, A - self.y)
        db = (1 / len(self.y)) * np.sum(A - self.y)
        return dW, db

    def update(self, dW, db, learning_rate):
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

    def train(self, n_iter=100, learning_rate=0.1):
        self.loss = []
        for i in range(n_iter):
            A = self.model(self.X)
            self.loss.append(self.log_loss(A))
            dW, db = self.gradients(A)
            self.update(dW, db, learning_rate)

        y_pred, _ = self.predict(self.X)
        self.accuracy = accuracy_score(self.y, y_pred)
        return self.W, self.b

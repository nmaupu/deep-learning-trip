from ml.model import LearningModel
from ml.neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    model = LearningModel(X, y)
    W, b = model.train(n_iter=100, learning_rate=0.1)

    print("Neuron has been trained.")
    print("W=", W)
    print("b=", b)
    print("accuracy score=", model.accuracy)

    new_plant = np.array([2, 1])

    x0 = np.linspace(-1, 4, 100)
    x1 = (-W[0] * x0 - b) / W[1]

    plt.plot(model.loss)
    plt.show()

    plt.scatter(model.X[:, 0], model.X[:, 1], c=model.y, cmap="summer")
    plt.scatter(new_plant[0], new_plant[1], c="red")
    plt.plot(x0, x1, c="orange")
    plt.show()

    neuron = Neuron(W, b)
    print(neuron.predict(new_plant))

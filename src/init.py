from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    n = Neuron(X, y)
    n.train(n_iter=100, learning_rate=0.1)

    print("Neuron has been trained.")
    print("W=", n.W)
    print("b=", n.b)
    print("accuracy score=", n.accuracy)

    new_plant = np.array([2, 1])

    x0 = np.linspace(-1, 4, 100)
    x1 = (-n.W[0] * x0 - n.b) / n.W[1]

    plt.scatter(n.X[:, 0], n.X[:, 1], c=n.y, cmap="summer")
    plt.scatter(new_plant[0], new_plant[1], c="red")
    plt.plot(x0, x1, c="orange")
    plt.show()

    print(n.predict(new_plant))

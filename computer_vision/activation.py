import numpy as np
from matplotlib import pyplot as plt


def sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = 1 / (1+np.exp(-x))
    plt.plot(x, y)
    plt.grid()
    plt.show()


def tanh():
    x = np.arange(-10, 10, 0.1)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    plt.plot(x, y)
    plt.grid()
    plt.show()


def relu():
    x = np.arange(-10, 10, 0.1)
    y = np.where(x<0, 0, x)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def leaky_relu():
    x = np.arange(-2, 2, 0.1)
    y = np.where(x<0, 0.01*x, x)
    plt.plot(x, y)
    plt.grid()
    plt.show()

leaky_relu()

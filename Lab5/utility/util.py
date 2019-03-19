import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_iris


def configure_plots():
    '''Configures plots by making some quality of life adjustments'''
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2
    
    
def load_toy():
    data = load_iris()
    y = data.target
    y[y > 0] = -1
    y[y == 0] = 1
    
    return data.data[:, :2], y


def optimize(gradient_fn, X, y, theta, eta=1e-3, iterations=1e4):
    '''
    computes weights W* that optimize the given the derivative of the loss function
    DFN given starting weights W
    '''
    for _ in range(int(iterations)):
        grad = gradient_fn(X, y, theta)
        theta -= eta * grad

    return theta

def sigmoid(x):
    return expit(x)

def logistic_gradient(X, y, theta):
    N = X.shape[0]
    return np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / N

def optimize_logistic(X, y, theta=None, **kwargs):
    if theta is None:
        _, d = X.shape
        theta = np.zeros(d)
    y = (y+1 != 0) *1 # make labels (0,1)
    return optimize(logistic_gradient, X, y, theta, iterations=5e5, **kwargs)

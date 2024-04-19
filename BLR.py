# Imports

import argparse
import random
import time
import numpy as np
import matplotlib
from scipy.stats import multivariate_normal

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init():
    # Variables

    n = int(args.n)
    sigma = 0.4 ** 2
    tau = 5.0 ** 2

    # Sampled observations

    x = []
    y = []

    # Init plot

    ax = plt.gca()

    # Sampling loop

    for i in range(n):
        # Clear previous plot

        plt.cla()

        # Add a new sample to the observations

        x_, y_ = data_generator(1, third_order_function, sigma)
        x.append(x_[0])
        y.append(y_[0])

        # Model selection

        d = model_selection(x, y, sigma, tau)
        # d = 4
        print('Selected polynomial order: {}'.format(d))

        # Compute posterior

        beta_hat, covar = posterior(poly_expansion(x, d), y, sigma, tau)

        # Evaluate regression function

        px = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution
              for i in range(args.plot_resolution)]
        pxx = poly_expansion(px, d)

        py, py_std = posterior_predictive(pxx, beta_hat, covar, sigma)

        # Uncertainty boundaries

        upper = (py + py_std).flatten().tolist()
        lower = (py - py_std).flatten().tolist()
        # Plot

        plt.axis(args.plot_boundaries)
        plt.fill_between(px, upper, lower, alpha=0.5, label='Standard deviation')
        plt.scatter(x, y, s=args.scatter_size, label='Observations')
        plt.plot(px, py, color='red', label='Polynomial estimate')

        ax.legend(loc='upper right')
        plt.rc('xtick')
        plt.rc('ytick')
        plt.title(args.title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)

    plt.show()


def third_order_function(x):
    """A third order polynomial function
    """

    return x ** 3 - x


def poly_expansion(x, d):
    """Polynomial expansion of vector x
    """

    xx = [[pow(xi, j) for j in range(0, d)] for xi in x]

    return np.matrix(xx)


def polynomial_value(x, beta):
    """Evaluate polynomial at data X, given weight parameters beta
    """

    return [sum([b * (xi ** j) for j, b in enumerate(beta)]) for i, xi in enumerate(x)]


def data_generator(n, f, sigma):
    """Generate training data
    """

    x = [3 * (random.random() - 0.5) for _ in range(n)]
    y = [np.random.normal(f(xi), sigma ** 0.5, 1)[0] for xi in x]

    return x, y


def posterior(X, y, sigma, tau):
    """Posterior
    """
    X = np.array(X)
    y = np.array(y)

    # TODO: Compute the mean vector (beta estimate)
    XX = np.matmul(X.T, X)
    inv = np.linalg.inv(XX + (sigma / tau) * np.identity(len(X[0])))
    beta = np.matmul(np.matmul(inv, X.T), y)

    # TODO: Compute the covariance matrix
    covar = sigma * inv

    return beta, covar


def posterior_predictive(X, beta_hat, covar, sigma):
    """Posterior predictive
    """
    y = []
    y_std = []
    for x in X:
        x = np.array(x).T
        beta_hat = np.array(beta_hat)

        mean = np.matmul(x.T, beta_hat)[0]
        variance = sigma + np.matmul(np.matmul(x.T, covar), x)[0][0]

        # TODO: Compute the y prediction given model parameters beta
        y.append(mean)

        # TODO: Compute the standard deviation
        y_std.append(np.sqrt(variance))

    y = np.array(y)
    y_std = np.array(y_std)

    return y, y_std


def model_selection(x, y, sigma, tau):
    """Model selection
    """

    n = len(y)
    d_min = 1
    d_max = 20
    score_list = []

    for d in range(d_min, d_max):
        xx = poly_expansion(x, d)

        # TODO: Predict y given x. Use the posterior and posterior_predictive methods.
        beta_hat, covar = posterior(xx, y, sigma, tau)
        py, py_std = posterior_predictive(xx, beta_hat, covar, sigma)

        mean = py.T
        variance = sigma * np.identity(n)
        # TODO: Compute the log likelihood, use the multivariate_normal method
        log_likelihood = multivariate_normal.logpdf(y, mean, variance)

        # TODO: Compute the BIC score
        score = -2 * log_likelihood + (d + 1) * np.log(n)
        score_list.append(score)

    # TODO: Select a proper degree of freedom: k + d_min
    d = np.argmin(score_list) + d_min

    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument('--title',
                        default='Ex5: Bayesian Linear Regression',
                        required=False)

    parser.add_argument('--n',
                        default=150,
                        required=False)

    parser.add_argument('--plot-boundaries',
                        default=[-1.5, 1.5, -1.5, 3],  # min_x, max_x, min_y, max_y
                        required=False)

    parser.add_argument('--plot-resolution',
                        default=100,
                        required=False)

    parser.add_argument('--scatter-size',
                        default=20,
                        required=False)

    args = parser.parse_args()

    init()

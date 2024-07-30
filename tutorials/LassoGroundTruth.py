from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

import os.path
from sklearn import datasets
from sklearn.linear_model import lasso_path


def lasso_ground_truth(X, y):

    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    n = 20
    variance = 0.7
    lambda_min_exp = -1
    lambda_max_exp = 5
    context_size = 1000
    uniform_lambdas = np.random.rand(context_size)
    lambdas_exp = 10**(uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp)
    lambdas_exp = lambdas_exp * variance / n
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=lambdas_exp, eps=eps)
    # print(alphas_lasso)

    # Display results

    plt.figure(1)
    colors = cycle(["b", "r", "g", "c", "k"])
    # neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_lasso = np.log10(alphas_lasso)

    l1 = []
    for index, (coef_l, c) in enumerate(zip(coefs_lasso, colors)):
        plt.plot(neg_log_alphas_lasso, coef_l)
        l1.append(index)

    plt.xlabel("alpha")
    plt.ylabel("coefficients")
    plt.title("Lasso")
    plt.legend(l1, loc="lower left")
    plt.axis("tight")
    folder_name = "./figures/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig("./figures/GroundTruth_lasso_beta_vs_log_alpha.pdf")

    plt.show()


if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)

    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
    lasso_ground_truth(X, y)

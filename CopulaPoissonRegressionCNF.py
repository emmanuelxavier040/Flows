import math

import numpy as np
import torch
import scipy as sp

import Utilities
import GroupPoissonRegressionCNF_withoutBetas
import Visualizations as View

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list, data_sample_size,
                                                         data_noise_sigma):
    X = torch.zeros((data_sample_size, dimension))
    W = torch.zeros((1, dimension))

    for group_index in range(len(grouped_indices_list)):
        g_size = len(grouped_indices_list[group_index])
        mean = torch.zeros(g_size)
        cov = 0.7 * torch.ones((g_size, g_size)) + 0.3 * torch.eye(g_size)
        mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
        x_samples = mvn_dist.sample(torch.Size([data_sample_size]))
        X[:, grouped_indices_list[group_index]] = x_samples

        if group_index % 2 == 0:
            W[:, grouped_indices_list[group_index]] = torch.randn(g_size)
        else:
            W[:, grouped_indices_list[group_index]] = torch.zeros(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    mean_poisson = torch.exp(Y)
    Z = torch.poisson(mean_poisson) + 1

    return X, Z, W, v, Y, mean_poisson


def generate_synthetic_data_with_group_correlation(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma):
    X = torch.zeros((data_sample_size, dimension))
    W = torch.zeros((1, dimension))
    num_samples = data_sample_size

    group_indices = grouped_indices_list[0]
    g_size = len(group_indices)
    mean = torch.zeros(g_size)
    cov = 0.7 * torch.ones((g_size, g_size)) + 0.3 * torch.eye(g_size)
    mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
    x_samples = mvn_dist.sample(torch.Size([data_sample_size]))

    noise = 0
    X[:, group_indices] = x_samples + noise
    W[:, group_indices] = torch.randn(g_size)

    group_indices = grouped_indices_list[1]
    g_size = len(group_indices)
    noise = 0.1
    X[:, group_indices] = x_samples + noise
    W[:, group_indices] = torch.randn(g_size)

    group_indices = grouped_indices_list[2]
    g_size = len(group_indices)
    noise = 0.5
    X[:, group_indices] = x_samples + noise
    W[:, group_indices] = torch.randn(g_size)

    group_indices = grouped_indices_list[3]
    g_size = len(group_indices)
    noise = 0.9
    mean = torch.zeros(g_size)
    cov = 0.7 * torch.ones((g_size, g_size)) + 0.3 * torch.eye(g_size)
    mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
    x_samples = mvn_dist.sample(torch.Size([data_sample_size]))
    X[:, group_indices] = x_samples + noise
    W[:, group_indices] = torch.randn(g_size)
    #
    # for i in range(2, len(grouped_indices_list)):
    #     group_index = i
    #     g_size = len(grouped_indices_list[group_index])
    #
    #     if i % 2 == 0:
    #         g3_base = torch.rand(num_samples, 1)
    #         X[:, grouped_indices_list[group_index]] = g3_base + torch.rand(num_samples, g_size)
    #     else:
    #         g2_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    #         X[:, grouped_indices_list[group_index]] = g2_base + torch.normal(mean=0, std=0.1,
    #                                                                          size=(num_samples, g_size))
    #     W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    mean_poisson = torch.exp(Y)
    Z = torch.poisson(mean_poisson) + 1
    return X, Z, W.squeeze(0)


def generate_regression_dataset(dimension, grouped_indices_list, data_sample_size,
                                                         data_noise_sigma):
    mean = torch.zeros(dimension)
    cov = 0.7 * torch.ones((dimension, dimension)) + 0.3 * torch.eye(dimension)
    mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
    X = mvn_dist.sample(torch.Size([data_sample_size]))
    W = torch.randn(dimension)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    mean_poisson = torch.exp(Y)
    Z = torch.poisson(mean_poisson) + 1

    return X, Z, W


def generate_copula_correlation_matrix(likelihood_sigma, X, covariance_matrix):
    variance = likelihood_sigma ** 2
    A = variance * covariance_matrix
    I = torch.eye(X.shape[0])
    term_2 = Utilities.woodbury_matrix_conversion(A, X.T, I, X, device)
    correlation_matrix = variance * Utilities.woodbury_matrix_conversion(I, X, -1 * term_2, X.T, device)
    return correlation_matrix


def generate_tau_covariance_matrix(tau_MAP, grouped_indices_list):
    flat_indices = [idx for sublist in grouped_indices_list for idx in sublist]
    flat_values = [value for value, sublist in zip(tau_MAP, grouped_indices_list) for _ in sublist]
    inverse_values = [1 / value for value in flat_values]
    max_index = max(flat_indices)
    matrix_size = max_index + 1
    covariance_matrix = torch.zeros((matrix_size, matrix_size), dtype=torch.float32)
    covariance_matrix[flat_indices, flat_indices] = torch.tensor(inverse_values)
    return covariance_matrix


def main():
    # Set the parameters
    epochs = 1000
    dimension = 12
    group_size = 3
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    zero_weight_group_index = 2
    data_sample_size = 50
    data_noise_sigma = 0.8
    likelihood_sigma = 2.0
    flow_sample_size = 1
    context_size = 1000
    lambda_min_exp = -3
    lambda_max_exp = 8
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, zero_weight_group_index:{zero_weight_group_index}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data_with_zero_group_coefficients(dimension,
    #                                                                                           grouped_indices_list,
    #                                                                                           data_sample_size,
    #                                                                                           data_noise_sigma)
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data(dimension, data_sample_size, data_noise_sigma)

    # X, Z, W = generate_synthetic_data_with_group_correlation(dimension, grouped_indices_list, data_sample_size,
    #                                                          data_noise_sigma)
    X, Z, W = generate_regression_dataset(dimension, grouped_indices_list, data_sample_size, data_noise_sigma)

    X /= X.std(0)

    train_ratio = 0.8
    X_train, Z_train, X_test, Z_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Z)

    X_torch = X_train.to(device)
    Z_torch = Z_train.to(device)

    flows, lambda_max_likelihood = GroupPoissonRegressionCNF_withoutBetas.posterior(X_train, Z_train, X_torch, Z_torch,
                                                                                    likelihood_sigma,
                                                                                    grouped_indices_list, epochs,
                                                                                    flow_sample_size, context_size,
                                                                                    lambda_min_exp, lambda_max_exp,
                                                                                    learning_rate, W)
    print("Best lambda: ", lambda_max_likelihood)
    uniform_lambdas = torch.rand(1).to(device)
    context = (uniform_lambdas * (lambda_max_likelihood - lambda_max_likelihood) + lambda_max_likelihood).view(-1, 1)
    flow_samples, flow_log_prob = flows.sample_and_log_prob(num_samples=1000, context=context)

    G = len(grouped_indices_list)
    tau_samples = flow_samples[:, :, : G]
    tau_MAP = tau_samples[0].mean(dim=0)

    covariance_matrix = generate_tau_covariance_matrix(tau_MAP, grouped_indices_list)
    copula_correlation_matrix = generate_copula_correlation_matrix(likelihood_sigma, X_train, covariance_matrix)
    print(copula_correlation_matrix)

    correlation_graph_title = "Poisson_Group_Lasso_Regression_Correlation_Matrix"
    View.plot_correlation_matrix(copula_correlation_matrix, correlation_graph_title)


if __name__ == "__main__":
    main()

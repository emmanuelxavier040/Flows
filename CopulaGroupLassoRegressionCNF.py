import os
import shutil

import numpy as np
import torch
import scipy as sp


import Utilities
import GroupLassoRegressionCNF_withoutBetas
import Visualizations as View
import GroupLassoRegressionCNF

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
        W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

        # if group_index % 2 == 0:
        #     W[:, grouped_indices_list[group_index]] = torch.randn(g_size)
        # else:
        #     W[:, grouped_indices_list[group_index]] = torch.zeros(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta

    return X, Y, W


def generate_synthetic_data_with_group_correlation(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma, col_replicate, group_replicate):
    X = torch.zeros((data_sample_size, dimension))
    W = torch.zeros((1, dimension))
    num_samples = data_sample_size

    group_indices = grouped_indices_list[0]
    g_size = len(group_indices)
    mean = torch.zeros(g_size)
    cov = 0.5 * torch.ones((g_size, g_size)) + 0.9 * torch.eye(g_size)
    mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
    x_samples = mvn_dist.sample(torch.Size([data_sample_size]))

    X[:, group_indices] = x_samples
    W[:, group_indices] = torch.randn(g_size)

    for i in range(0, group_replicate):
        group_indices = np.array(grouped_indices_list[i+1])
        g_size = len(group_indices)
        noise_sigma = group_noise_sigma
        col_to_replicate = np.arange(0, col_replicate, dtype=int)
        col_to_random = np.arange(col_replicate, g_size, dtype=int)
        noise = torch.randn(data_sample_size) * torch.tensor(noise_sigma ** 2)
        X[:, group_indices[col_to_replicate]] = x_samples[:, col_to_replicate] + noise.unsqueeze(-1)
        X[:, group_indices[col_to_random]] = torch.rand(num_samples, 1) + torch.rand(num_samples, len(col_to_random))
        W[:, group_indices] = torch.randn(g_size)

    for i in range(group_replicate+1, len(grouped_indices_list)):
        group_index = i
        g_size = len(grouped_indices_list[group_index])
        X[:, grouped_indices_list[group_index]] = torch.rand(num_samples, 1) + torch.rand(num_samples, g_size)
        W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    return X, Y, W.squeeze(0)


def generate_regression_dataset(n_samples, n_features, n_non_zero, noise_std):
    assert n_features >= n_non_zero

    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    coefficients[non_zero_indices] = np.random.normal(0, 1, n_non_zero)

    scale_matrix = np.eye(n_features)
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std ** 2,
                                                   n_samples)

    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(coefficients).float()


def generate_copula_correlation_matrix(likelihood_sigma, X, covariance_matrix):
    variance = likelihood_sigma ** 2
    A = variance * covariance_matrix
    I = torch.eye(X.shape[0])
    term_2 = Utilities.woodbury_matrix_conversion(A, X.T, I, X, device)
    correlation_matrix = variance * Utilities.woodbury_matrix_conversion(I, X, -1 * term_2, X.T, device)
    return correlation_matrix


def generate_tau_covariance_matrix(tau_MAP, grouped_indices_list):
    print("Tau MAP : ", tau_MAP)
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
    epochs = 2000
    dimension = 16
    group_size = 4
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    zero_weight_group_index = 2
    data_sample_size = 100
    data_noise_sigma = 0.8
    likelihood_sigma = 2.0
    flow_sample_size = 1
    context_size = 1000
    lambda_min_exp = -3
    lambda_max_exp = 6
    learning_rate = 1e-3

    # print(f"============= Parameters ============= \n"
    #       f"Dimension:{dimension}, zero_weight_group_index:{zero_weight_group_index}, "
    #       f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data_with_zero_group_coefficients(dimension,
    #                                                                                           grouped_indices_list,
    #                                                                                           data_sample_size,
    #                                                                                           data_noise_sigma)
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data(dimension, data_sample_size, data_noise_sigma)

    # X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    #

    # X, Y, W = generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list, data_sample_size,
    #                                                                data_noise_sigma)

    n_groups = int(dimension / group_size)
    group_replicate_list = np.arange(1, n_groups, 1).tolist()
    col_replicate_list = np.arange(1, group_size+1, 1).tolist()
    group_noise_list = np.arange(0, 1, 0.5)

    for group_replicate in group_replicate_list:
        for col_replicate in col_replicate_list:
            for group_noise_sigma in group_noise_list:

                # group_noise_sigma = 2
                # col_replicate = group_size
                # group_replicate = 0
                # for group_noise_sigma in group_noise_list:
                X, Y, W = generate_synthetic_data_with_group_correlation(dimension, grouped_indices_list, data_sample_size,
                                                                         data_noise_sigma, group_noise_sigma, col_replicate, group_replicate)
                X /= X.std(0)

                train_ratio = 0.8
                X_train, Y_train, X_test, Y_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Y)

                X_torch = X_train.to(device)
                Y_torch = Y_train.to(device)

                flows, lambda_max_likelihood = GroupLassoRegressionCNF_withoutBetas.posterior(X_train, Y_train, X_torch, Y_torch,
                                                                                                likelihood_sigma,
                                                                                                grouped_indices_list, epochs,
                                                                                                flow_sample_size, context_size,
                                                                                                lambda_min_exp, lambda_max_exp,
                                                                                                learning_rate, W)
                # GroupLassoRegressionCNF.posterior(X_train.detach().cpu().numpy(), Y_train.detach().cpu().numpy(), X_torch, Y_torch, likelihood_sigma,
                #           grouped_indices_list, epochs, flow_sample_size, context_size,
                #           lambda_min_exp, lambda_max_exp, learning_rate, W)

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

                correlation_graph_title = "Group_Lasso_Regression_"+str(group_noise_sigma)
                View.plot_correlation_matrix(copula_correlation_matrix, correlation_graph_title)

                destination_directory = f'./figures/CExp_d{dimension}_n{data_sample_size}_G{dimension/(group_size)}_noise{group_noise_sigma}_gr_repl{group_replicate}_col_repl{col_replicate}'
                if not os.path.exists(destination_directory):
                    os.mkdir(destination_directory)

                source_directory = "./figures/"
                files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

                for file in files:
                    shutil.move(os.path.join(source_directory, file), destination_directory)


if __name__ == "__main__":
    main()

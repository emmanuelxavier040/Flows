import os
import shutil

import numpy as np
import torch
import scipy as sp


import Utilities
import GroupLassoRegressionCNF_withoutBetas as GL_taus
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


def generate_synthetic_data_with_group_correlation_2(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma, col_replicate, group_replicate):
    X = torch.zeros((data_sample_size, dimension))
    W = torch.zeros((1, dimension))
    num_samples = data_sample_size

    group_indices = grouped_indices_list[0]
    g_size = len(group_indices)
    mean = torch.zeros(g_size)
    scale_matrix = np.eye(g_size)
    covariance = sp.stats.wishart(df=g_size, scale=scale_matrix).rvs(1)
    covariance = torch.from_numpy(covariance).float()
    mvn_dist = torch.distributions.MultivariateNormal(mean, covariance)
    x_samples = mvn_dist.sample(torch.Size([data_sample_size]))

    X[:, group_indices] = x_samples
    W[:, group_indices] = torch.randn(g_size)

    for i in range(0, group_replicate):
        group_indices = np.array(grouped_indices_list[i+1])
        g_size = len(group_indices)
        noise_sigma = group_noise_sigma
        col_to_replicate = np.arange(0, col_replicate, dtype=int)
        noise = torch.randn(data_sample_size) * torch.tensor(noise_sigma ** 2)
        X[:, group_indices[col_to_replicate]] = x_samples[:, col_to_replicate] + noise.unsqueeze(-1)

        # Set random numbers to the rest of the columns in the data group
        # col_to_random = np.arange(col_replicate, g_size, dtype=int)
        # X[:, group_indices[col_to_random]] = torch.rand(num_samples, 1) + torch.rand(num_samples, len(col_to_random))

        # Set correlated data to rest of the columns in the data group
        mean = torch.ones(g_size)
        cov = 0.8 * torch.ones((g_size, g_size)) + torch.eye(g_size)
        mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
        x_samples = mvn_dist.sample(torch.Size([data_sample_size]))
        X[:, group_indices] = x_samples + torch.rand(num_samples, g_size)

        W[:, group_indices] = torch.randn(g_size)

    for i in range(group_replicate+1, len(grouped_indices_list)):
        group_index = i
        g_size = len(grouped_indices_list[group_index])

        # Set correlated data to rest of the groups
        mean = torch.ones(g_size)
        cov = 0.8 * torch.ones((g_size, g_size)) + torch.eye(g_size)
        mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
        x_samples = mvn_dist.sample(torch.Size([data_sample_size]))
        X[:, grouped_indices_list[group_index]] = x_samples + torch.rand(num_samples, g_size)
        W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

        # group_index = i
        # g_size = len(grouped_indices_list[group_index])
        # X[:, grouped_indices_list[group_index]] = torch.rand(num_samples, 1) + torch.rand(num_samples, g_size)
        # W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    return X, Y, W.squeeze(0)


def generate_synthetic_data_with_group_correlation_3(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma, col_replicate, group_replicate):
    d = 4 #Group_size
    cov_group1 = torch.eye(d) * 0.9 + 0.1 * torch.ones(d, d)
    cov_group2 = torch.eye(d) * 0.8 + 0.2 * torch.ones(d, d)
    inter_correlation = 0.5
    full_covariance = torch.zeros(2 * d, 2 * d)
    full_covariance[:d, :d] = cov_group1
    full_covariance[d:, d:] = cov_group2
    full_covariance[0, d + 1] = inter_correlation
    full_covariance[d + 1, 0] = inter_correlation

    mean = torch.zeros(2 * d)
    multivariate_normal_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=full_covariance)
    X = multivariate_normal_dist.sample(torch.Size([data_sample_size]))
    W = torch.randn(dimension)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    return X, Y, W.squeeze(0)


def save_data_covariance(covariance):
    if covariance.ndimension() != 2:
        raise ValueError("Input tensor must be 2-dimensional")
    array = covariance.numpy()
    format_str = "{: .4f}"
    f = open(f"./figures/Data_Covariance.txt", "a")
    for row in array:
        formatted_row = " ".join(format_str.format(val) for val in row)
        f.write(f"{formatted_row}\n")
    f.write(f"====================================================\n")
    f.close()


def print_data_covariance(covariance):
    if covariance.ndimension() != 2:
        raise ValueError("Input tensor must be 2-dimensional")
    array = covariance.numpy()
    format_str = "{: .4f}"
    for row in array:
        formatted_row = " ".join(format_str.format(val) for val in row)
        print(f"{formatted_row}\n")
    print(f"====================================================\n")


def nearest_positive_definite(matrix):
    # Find the nearest positive definite matrix
    sym_matrix = (matrix + matrix.t()) / 2
    eigvals, eigvecs = torch.linalg.eigh(sym_matrix)
    eigvals[eigvals < 1e-6] = 1e-6
    positive_def_matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
    return positive_def_matrix


def generate_synthetic_data_with_group_correlation_4(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma, col_replicate, group_replicate):
    group_covariances = []
    for indices in grouped_indices_list:
        d = len(indices)
        A = torch.randn(d, d)
        cov_matrix = torch.mm(A, A.t())
        cov_matrix += torch.eye(d) * 1e-4
        group_covariances.append(cov_matrix)

    full_covariance = torch.zeros(dimension, dimension)
    for i, indices in enumerate(grouped_indices_list):
        for j, idx1 in enumerate(indices):
            for k, idx2 in enumerate(indices):
                full_covariance[idx1, idx2] = group_covariances[i][j, k]

    group0_indices = grouped_indices_list[0]

    for g in range(1, group_replicate + 1):
        correlated_group_indices = grouped_indices_list[g]
        num_correlations = min(col_replicate, len(group0_indices), len(correlated_group_indices))

        for i in range(num_correlations):
            inter_correlation = 0.5
            full_covariance[group0_indices[i], correlated_group_indices[i]] = inter_correlation
            full_covariance[correlated_group_indices[i], group0_indices[i]] = inter_correlation

    full_covariance = nearest_positive_definite(full_covariance)
    save_data_covariance(full_covariance)
    cov = full_covariance.cpu().numpy()
    np.allclose(cov, cov.T)
    np.linalg.eig(cov)
    np.linalg.cholesky(cov)
    print("Good Covariance for Data!!")

    mean = torch.zeros(dimension)
    multivariate_normal_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=full_covariance)
    X = multivariate_normal_dist.sample(torch.Size([data_sample_size]))

    W = torch.randn(dimension)
    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    return X, Y, W.squeeze(0)


def generate_synthetic_data_with_group_correlation_5(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma, col_replicate, group_replicate):
    group_covariances = []
    strength = 0.7
    for indices in grouped_indices_list:
        d = len(indices)
        A = torch.randn(d, d)
        # A = torch.eye(d) * (1 - strength) + torch.ones(d, d) * strength
        cov_matrix = torch.mm(A, A.t())
        cov_matrix += torch.eye(d) * 1e-4
        group_covariances.append(cov_matrix)

    full_covariance = torch.zeros(dimension, dimension)
    for i, indices in enumerate(grouped_indices_list):
        for j, idx1 in enumerate(indices):
            for k, idx2 in enumerate(indices):
                full_covariance[idx1, idx2] = group_covariances[i][j, k]

    group0_indices = grouped_indices_list[0]

    correlated_group_indices_list = [group0_indices]
    for g in range(1, group_replicate + 1):
        correlated_group_indices_list.append(grouped_indices_list[g])

    for i in range(col_replicate):
        for g1 in range(len(correlated_group_indices_list)):
            for g2 in range(g1 + 1, len(correlated_group_indices_list)):
                group1_idx = correlated_group_indices_list[g1][i]
                group2_idx = correlated_group_indices_list[g2][i]

                start_row = correlated_group_indices_list[g2][0]
                end_row = group2_idx
                col = group1_idx
                inter_correlation = 1.5
                full_covariance[start_row:end_row + 1, col] = inter_correlation
                full_covariance[col, start_row:end_row + 1] = inter_correlation

                start_col = correlated_group_indices_list[g1][0]
                end_col = group1_idx
                row = group2_idx
                full_covariance[row, start_col:end_col+1] = inter_correlation
                full_covariance[start_col:end_col+1, row] = inter_correlation


    print_data_covariance(full_covariance)
    full_covariance = nearest_positive_definite(full_covariance)
    save_data_covariance(full_covariance)
    cov = full_covariance.cpu().numpy()
    np.allclose(cov, cov.T)
    np.linalg.eig(cov)
    np.linalg.cholesky(cov)
    print("Good Covariance for Data!!")

    mean = torch.zeros(dimension)
    multivariate_normal_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=full_covariance)
    X = multivariate_normal_dist.sample(torch.Size([data_sample_size]))

    W = torch.randn(dimension)
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
    # term_2 = Utilities.woodbury_matrix_conversion(A, X.T, I, X, device)
    # correlation_matrix = variance * Utilities.woodbury_matrix_conversion(I, X, -1 * term_2, X.T, device)

    correlation_matrix = variance * torch.inverse(I - Utilities.new_woodbury_identity(X, A, X.T, I, X, X.T))
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
    dimension = 9
    group_size = 3
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    data_sample_size = 100
    data_noise_sigma = 0.8
    likelihood_sigma = 2.0
    flow_sample_size = 1
    context_size = 1000
    lambda_min_exp = -3
    lambda_max_exp = 6
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension},"
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")
    # X, Z, W = generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list,
    #                                                                data_sample_size, data_noise_sigma)

    # X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    #

    # X, Y, W = generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list, data_sample_size,
    #                                                                data_noise_sigma)

    n_groups = int(dimension / group_size)
    group_replicate_list = np.arange(2, n_groups, 1).tolist()
    col_replicate_list = np.arange(2, group_size+1, 1).tolist()
    group_noise_list = np.arange(0, 1, 0.5)

    for group_replicate in group_replicate_list:
        for col_replicate in col_replicate_list:
            for group_noise_sigma in group_noise_list:

                iter_identifier = f"d{dimension}_n{data_sample_size}_G{dimension/(group_size)}_noise{group_noise_sigma}_gr_repl{group_replicate}_col_repl{col_replicate}"

                X, Y, W = generate_synthetic_data_with_group_correlation_5(dimension, grouped_indices_list, data_sample_size,
                                                                         data_noise_sigma, group_noise_sigma, col_replicate, group_replicate)
                X /= X.std(0)

                train_ratio = 0.8
                X_train, Y_train, X_test, Y_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Y)

                X_torch = X_train.to(device)
                Y_torch = Y_train.to(device)

                tau_flows, lambda_max_likelihood_taus = GL_taus.posterior(X_train, Y_train, X_torch, Y_torch,
                                                                                                likelihood_sigma,
                                                                                                grouped_indices_list, epochs,
                                                                                                flow_sample_size, context_size,
                                                                                                lambda_min_exp, lambda_max_exp,
                                                                                                learning_rate, W)
                beta_flows, lambda_max_likelihood_beta = GroupLassoRegressionCNF.posterior(X_train.detach().cpu().numpy(),
                                                                                           Y_train.detach().cpu().numpy(),
                                                                                           X_torch, Y_torch, likelihood_sigma,
                                                                                           grouped_indices_list, epochs,
                                                                                           flow_sample_size, context_size,
                                                                                           lambda_min_exp, lambda_max_exp,
                                                                                           learning_rate, W)

                GL_taus.experiment_compare_beta_path_from_taus_with_beta_path_group_lasso_for_lambda_range(dimension, X_torch, Y_torch,
                                                                                       likelihood_sigma, beta_flows,
                                                                                       tau_flows,
                                                                                       grouped_indices_list,
                                                                                       lambda_min_exp,
                                                                                       lambda_max_exp, context_size=1000,
                                                                                        text=iter_identifier)

                print("Best lambda: ", lambda_max_likelihood_taus)
                uniform_lambdas = torch.rand(1).to(device)
                context = (uniform_lambdas * (lambda_max_likelihood_taus - lambda_max_likelihood_taus) + lambda_max_likelihood_taus).view(-1, 1)
                flow_samples, flow_log_prob = tau_flows.sample_and_log_prob(num_samples=1000, context=context)

                G = len(grouped_indices_list)
                tau_samples = flow_samples[:, :, : G]
                tau_MAP = tau_samples[0].mean(dim=0)

                covariance_matrix = generate_tau_covariance_matrix(tau_MAP, grouped_indices_list)
                copula_correlation_matrix = generate_copula_correlation_matrix(likelihood_sigma, X_train, covariance_matrix)
                print(copula_correlation_matrix)

                correlation_graph_title = "Group_Lasso_Regression_"+str(group_noise_sigma)
                View.plot_correlation_matrix(copula_correlation_matrix, correlation_graph_title)

                destination_directory = f'./figures/CExp_'+iter_identifier
                if not os.path.exists(destination_directory):
                    os.mkdir(destination_directory)

                source_directory = "./figures/"
                files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

                for file in files:
                    shutil.move(os.path.join(source_directory, file), destination_directory)


if __name__ == "__main__":
    main()

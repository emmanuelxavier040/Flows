import math

import numpy as np
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn import Sin
from enflows.nn.nets import ResidualNet
from enflows.transforms import iResBlock
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import ConditionalSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.distributions import ConditionalDiagonalNormal
from torch import optim
import scipy as sp

import Visualizations as View
import GroupLassoRegressionCNF
import Utilities

from enflows.transforms.nonlinearities import Softplus

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def generate_tau_diagonal_matrices(tau_samples, grouped_indices_list):
    max_index = max(max(sublist) for sublist in grouped_indices_list) + 1
    repeat_counts = torch.tensor([len(sublist) for sublist in grouped_indices_list]).to(device)
    values_repeated = tau_samples.repeat_interleave(repeat_counts, dim=-1)
    flat_indices = [idx for sublist in grouped_indices_list for idx in sublist]
    flat_indices_tensor = torch.tensor(flat_indices).to(device)
    batch_size, depth, _ = tau_samples.shape
    list_tensors = torch.zeros((batch_size, depth, max_index)).to(device)
    list_tensors[:, :, flat_indices_tensor] = values_repeated.to(device)
    taus_diagonal_matrices = torch.diag_embed(1 / list_tensors)
    return taus_diagonal_matrices


def log_posterior_unnormalized(constants, Λ, grouped_indices_list, tau_samples, X, Y, lambdas_exp):
    c_1, gamma_term, yTΛy, c_4_JT, XTΛX = constants
    log_posterior = c_1 + gamma_term + yTΛy

    n = len(X)
    log_posterior = log_posterior + (n * -0.5) * torch.log(2 * torch.tensor(torch.pi))

    n_groups = len(grouped_indices_list)
    log_posterior = log_posterior + -0.5 * torch.sum(torch.log(tau_samples), dim=-1)

    taus_diagonal_matrices = generate_tau_diagonal_matrices(tau_samples, grouped_indices_list)

    A = (XTΛX + taus_diagonal_matrices)
    term_3 = -0.5 * torch.linalg.slogdet(A)[1]
    log_posterior = log_posterior + term_3

    ## A_inverse = torch.inverse(A)
    # A_inverse = Utilities.woodbury_matrix_conversion(taus_diagonal_matrices, X.T, Λ, X, device)
    # term_4 = 0.5 * torch.matmul(torch.matmul(c_4_JT, A_inverse), c_4_JT.T)
    #
    m = torch.matmul
    inv = Utilities.new_woodbury_identity(X, taus_diagonal_matrices, X.T, Λ, X, X.T, device)
    term_4 = 0.5 * m(m(m(m(Y.T, Λ), inv), Λ), Y).to(device)

    log_posterior = log_posterior + term_4

    n_features = len(X[0])
    lambda_list = 10 ** lambdas_exp

    term_5 = (- (lambda_list ** 2) / 2) * tau_samples.sum(dim=-1)
    log_posterior = log_posterior + term_5

    term_6 = 0.5 * (n_features + n_groups) * torch.log(0.5 * (lambda_list ** 2))
    log_posterior = log_posterior + term_6

    return log_posterior


def precompute_constants(grouped_indices_list, X, Y, likelihood_cov_matrix):
    Λ = torch.inverse(likelihood_cov_matrix).to(device)
    c_1 = -0.5 * torch.linalg.slogdet(torch.inverse(Λ))[1]
    gamma_term = 0
    for group_indices in grouped_indices_list:
        gamma_term = gamma_term - torch.lgamma(0.5 * torch.tensor(len(group_indices) + 1))

    yTΛy = -0.5 * torch.matmul(torch.matmul(Y.T, Λ), Y)

    c_4_JT = torch.matmul(torch.matmul(Y.T, Λ), X).to(device)
    XTΛX = torch.matmul(torch.matmul(X.T, Λ), X).to(device)

    return c_1.to(device), gamma_term.to(device), yTΛy, c_4_JT, XTΛX


def train_CNF(flows, d, grouped_indices_list, X, Y, X_torch, Y_torch, likelihood_sigma, epochs, tau_sample_size,
              context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
    file_name = f'CNF_d{d}_n{tau_sample_size}_e{epochs}_lmin{lambda_min_exp}_lmax{lambda_max_exp}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)

    print("Starting training the flows")
    losses = []
    lambda_max_likelihood = -math.inf

    T0 = 5.0
    Tn = 0.01
    cool_step_iteration = 200
    cool_num_iter = epochs // cool_step_iteration

    def cooling_function(t):
        if t < (cool_num_iter - 1):
            k = t / (cool_num_iter - 1)
            alpha = Tn / T0
            return T0 * (alpha ** k)
        else:
            return Tn

    try:
        n_data_samples = len(X_torch)
        likelihood_cov_matrix = (likelihood_sigma ** 2) * torch.eye(n_data_samples).to(device)
        Λ = torch.inverse(likelihood_cov_matrix).to(device)

        constants = precompute_constants(grouped_indices_list, X_torch, Y_torch, likelihood_cov_matrix)
        for epoch in range(epochs):
            t = epoch // (epochs / cool_num_iter)
            T = cooling_function(t=t)

            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = lambdas_exp
            tau_samples, tau_log_prob = flows.sample_and_log_prob(num_samples=tau_sample_size, context=context)
            log_p = log_posterior_unnormalized(constants, Λ, grouped_indices_list, tau_samples, X_torch, Y_torch,
                                               lambdas_exp)
            log_p = torch.clamp(log_p, min=-1e10, max=1e10)

            loss = torch.mean(tau_log_prob - (log_p / T))
            # loss = torch.mean(tau_log_prob - log_p)
            loss.backward()

            if epoch % 10 == 0 or epoch + 1 == epochs:
                if epoch % cool_step_iteration == 0:
                    print("Temperature: ", T)
                print("Loss after iteration {}: ".format(epoch), loss.tolist())
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(flows.parameters(), 1)
            optimizer.step()

            next_T = cooling_function((epoch + 1) // (epochs / cool_num_iter))
            if next_T < 1 <= T or (T == 1. and epoch + 1 == epochs):
                lambdas_sorted, q_samples_sorted, losses_sorted = sample_from_flow_for_plots(flows,
                                                                                             grouped_indices_list,
                                                                                             X_torch, Y_torch,
                                                                                             likelihood_sigma, 100, 100,
                                                                                             lambda_min_exp,
                                                                                             lambda_max_exp)

                log_likelihood_means = np.mean(-losses_sorted, axis=1)
                lambda_max_likelihood = lambdas_sorted[np.argmax(log_likelihood_means)]

                solution_type = "Group-Lasso-Solution Path"
                View.plot_flow_group_coefficients_path_vs_ground_truth(X, Y, lambdas_sorted,
                                                                       q_samples_sorted, solution_type)

                title = "GL-Without_betas_Log_marginal_likelihood"
                View.plot_log_marginal_likelihood_vs_lambda(X, Y, lambdas_sorted, losses_sorted, likelihood_sigma ** 2,
                                                            title)

    except KeyboardInterrupt:
        print("interrupted..")

    # save_model(flows, file_name)

    return flows, losses, lambda_max_likelihood


def generate_synthetic_data(d, grouped_indices_list, zero_weight_group_index, n, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X and Y
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))

    num_data_samples = torch.Size([n])
    num_samples = n
    X = torch.zeros((num_samples, d))

    g1_size = len(grouped_indices_list[0])
    # g1_mean = torch.randn(num_samples, 1)
    g1_mean = torch.normal(mean=10, std=3, size=(num_samples, g1_size))
    X[:, grouped_indices_list[0]] = g1_mean + torch.normal(mean=0, std=0.1, size=(num_samples, g1_size))

    g2_size = len(grouped_indices_list[1])
    # g2_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    g2_base = torch.normal(mean=10, std=3, size=(num_samples, g2_size))
    X[:, grouped_indices_list[1]] = g2_base + torch.normal(mean=0, std=0.1, size=(num_samples, g2_size))

    g3_size = len(grouped_indices_list[2])
    # g3_base = torch.rand(num_samples, 1)
    # X[:, grouped_indices_list[2]] = g3_base + torch.rand(num_samples, g3_size)
    g3_base = torch.normal(mean=10, std=3, size=(num_samples, g3_size))
    X[:, grouped_indices_list[2]] = g3_base + torch.normal(mean=0, std=0.1, size=(num_samples, g3_size))

    ##### =============================================================================

    # g4_size = len(grouped_indices_list[3])
    # g4_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    # X[:, grouped_indices_list[3]] = g4_base + torch.normal(mean=2, std=0.5, size=(num_samples, g4_size))
    #
    # g5_size = len(grouped_indices_list[4])
    # g5_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    # X[:, grouped_indices_list[4]] = g5_base + torch.normal(mean=4.5, std=0.1, size=(num_samples, g5_size))

    # W = torch.rand(d) * 20 - 10
    W = torch.randn(d)

    min_val = torch.min(W)
    max_val = torch.max(W)
    W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W.shape)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    W[grouped_indices_list[zero_weight_group_index]] = 0

    v = torch.tensor(noise ** 2)
    delta = torch.randn(num_data_samples) * v
    # delta = torch.normal(0, noise ** 2, num_data_samples)
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


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


def generate_synthetic_data_with_group_correlation(dimension, grouped_indices_list, data_sample_size,
                                                   data_noise_sigma, group_noise_sigma):
    X = torch.zeros((data_sample_size, dimension))
    W = torch.zeros((1, dimension))
    num_samples = data_sample_size

    group_indices = grouped_indices_list[0]
    g_size = len(group_indices)
    mean = torch.zeros(g_size)
    cov = 0.1 * torch.ones((g_size, g_size)) + 0.5 * torch.eye(g_size)
    mvn_dist = torch.distributions.MultivariateNormal(mean, cov)
    x_samples = mvn_dist.sample(torch.Size([data_sample_size]))

    noise = 0
    X[:, group_indices] = x_samples + noise
    W[:, group_indices] = torch.randn(g_size)

    group_indices = grouped_indices_list[1]
    g_size = len(group_indices)
    noise_sigma = group_noise_sigma
    noise = torch.randn(data_sample_size) * torch.tensor(noise_sigma ** 2)
    X[:, group_indices] = x_samples + noise.unsqueeze(-1)
    W[:, group_indices] = torch.randn(g_size)

    # group_indices = grouped_indices_list[2]
    # g_size = len(group_indices)
    # noise_sigma = group_noise_sigma
    # noise = torch.randn(data_sample_size) * torch.tensor(noise_sigma ** 2)
    # X[:, group_indices] = x_samples + noise.unsqueeze(-1)
    # W[:, group_indices] = torch.randn(g_size)
    #
    # group_indices = grouped_indices_list[3]
    # g_size = len(group_indices)
    # noise = 0
    # X[:, group_indices] = x_samples + noise
    # W[:, group_indices] = torch.randn(g_size)

    for i in range(2, len(grouped_indices_list)):
        group_index = i
        g_size = len(grouped_indices_list[group_index])

        if i % 2 == 0:
            g3_base = torch.rand(num_samples, 1)
            X[:, grouped_indices_list[group_index]] = g3_base + torch.rand(num_samples, g_size)
        else:
            g3_base = torch.rand(num_samples, 1)
            X[:, grouped_indices_list[group_index]] = g3_base + torch.rand(num_samples, g_size)
            # g2_base = torch.distributions.Exponential(1).sample((num_samples, 1))
            # X[:, grouped_indices_list[group_index]] = g2_base + torch.normal(mean=0, std=0.1,
            #                                                                  size=(num_samples, g_size))
        W[:, grouped_indices_list[group_index]] = torch.randn(g_size)

    v = torch.tensor(data_noise_sigma ** 2)
    delta = torch.randn(data_sample_size) * v
    Y = torch.matmul(X, W.unsqueeze(-1)).squeeze(-1).squeeze(0) + delta
    return X, Y, W.squeeze(0)


def build_sum_of_sigmoid_conditional_flow_model(d):
    context_features = 16
    hidden_features = 64
    num_layers = 3

    # context_features = 32
    # hidden_features = 128
    # num_layers = 10

    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    for _ in range(num_layers):
        transforms.append(
            InverseTransform(
                ConditionalSumOfSigmoidsTransform(
                    features=d, hidden_features=hidden_features,
                    context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )
        transforms.append(
            InverseTransform(
                ActNorm(features=d)
            )
        )

    transforms.append(InverseTransform(Softplus(eps=1e-6)))
    # transforms.append(InverseTransform(Exp()))

    transforms = transforms[::-1]
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=hidden_features,
                                num_blocks=3, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def sample_from_flow_for_plots(flows, grouped_indices_list, X, Y, likelihood_sigma, context_size, flow_sample_size,
                               lambda_min_exp, lambda_max_exp):
    num_iter = 10
    lambdas, flow_samples_list, losses = [], [], []

    with torch.no_grad():
        n_data_samples = len(X)
        likelihood_cov_matrix = (likelihood_sigma ** 2) * torch.eye(n_data_samples)
        Λ = torch.inverse(likelihood_cov_matrix).to(device)

        constants = precompute_constants(grouped_indices_list, X, Y, likelihood_cov_matrix)
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            q_samples, q_log_probs = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_p_samples = log_posterior_unnormalized(constants, Λ, grouped_indices_list, q_samples, X, Y, lambdas_exp)
            loss = q_log_probs - log_p_samples

            lambdas.append((10 ** lambdas_exp).squeeze().cpu().detach().numpy())
            flow_samples_list.append(q_samples.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

    flow_samples_list, lambdas, losses = (np.concatenate(flow_samples_list, 0),
                                          np.concatenate(lambdas, 0), np.concatenate(losses, 0))
    lambda_sort_order = lambdas.argsort()

    lambdas_sorted = lambdas[lambda_sort_order]
    q_samples_sorted = flow_samples_list[lambda_sort_order]
    losses_sorted = losses[lambda_sort_order]
    return lambdas_sorted, q_samples_sorted, losses_sorted


def posterior(X, Y, X_torch, Y_torch, likelihood_sigma, grouped_indices_list, epochs, tau_sample_size,
              context_size, lambda_min_exp, lambda_max_exp, learning_rate, W):
    n_groups = len(grouped_indices_list)
    dimension = n_groups
    original_W = W.tolist()
    print("Original Parameters: ", original_W)
    fixed_lambda_exp = torch.rand(1)
    print("Fixed Lambda exponent: ", fixed_lambda_exp)

    # ==================================================================
    # train conditional flows

    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    # flows = build_flow(dimension)
    flows.to(device)

    flows, losses, lambda_max_likelihood = train_CNF(flows, dimension, grouped_indices_list, X, Y, X_torch, Y_torch,
                                                     likelihood_sigma, epochs,
                                                     tau_sample_size,
                                                     context_size, lambda_min_exp, lambda_max_exp,
                                                     learning_rate)

    # View.plot_loss(losses)
    solution_type = "No_Beta_Group-Lasso-MAP"
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_from_flow_for_plots(flows, grouped_indices_list,
                                                                                 X_torch, Y_torch,
                                                                                 likelihood_sigma, 100, 100,
                                                                                 lambda_min_exp, lambda_max_exp)

    View.plot_flow_group_coefficients_path_vs_ground_truth(X, Y, lambdas_sorted, q_samples_sorted, solution_type)
    return flows, lambda_max_likelihood


def generate_group_indices(grouped_indices_list):
    total_indices = len(set(index for sublist in grouped_indices_list for index in sublist))

    group_indices = [0] * total_indices

    for group_id, indices in enumerate(grouped_indices_list):
        for index in indices:
            group_indices[index] = group_id

    return group_indices


def calculate_beta_distribution_by_plugging_in_taus_in_main_equation(likelihood_sigma, X_train, Y_train,
                                                                     taus_diagonal_matrix):
    likelihood_cov_matrix = (likelihood_sigma ** 2) * torch.eye(X_train.shape[0]).to(device)
    Λ = torch.inverse(likelihood_cov_matrix).to(device)
    XTΛX = torch.matmul(torch.matmul(X_train.T, Λ), X_train).to(device)
    cov_beta_dist = XTΛX + taus_diagonal_matrix
    m = torch.matmul

    # cov2_1 = Utilities.woodbury_matrix_conversion(taus_diagonal_matrix, X_train.t(), Λ, X_train, device)
    # mean_beta_dist = m(m(m(cov2_1, X_train.t()), Λ.t()), Y_train)

    inv_XT = Utilities.woodbury_identity_special(taus_diagonal_matrix, X_train.T, Λ, X_train, X_train.T, device)
    mean_beta_dist = m(m(inv_XT, Λ.t()), Y_train)
    return mean_beta_dist, cov_beta_dist


def experiment_compare_betas_from_group_lasso_for_a_lambda(dimension, X_train, Y_train, likelihood_sigma, beta_flows, tau_flows,
                                                           grouped_indices_list, lambda_max_likelihood):
    context_size = 1
    num_samples = 100
    # num_samples = 1000
    uniform_lambdas = torch.rand(context_size).to(device)
    context = (uniform_lambdas * (lambda_max_likelihood - lambda_max_likelihood) + lambda_max_likelihood).view(-1, 1)
    tau_samples, tau_log_prob = tau_flows.sample_and_log_prob(num_samples=num_samples, context=context)

    taus_diagonal_matrices = generate_tau_diagonal_matrices(tau_samples, grouped_indices_list)

    beta_for_tau = []
    for tau_sample, taus_diagonal_matrix in zip(tau_samples[0], taus_diagonal_matrices[0]):
        mean2, cov2 = calculate_beta_distribution_by_plugging_in_taus_in_main_equation(likelihood_sigma,
                                                                                    X_train, Y_train,
                                                                                    taus_diagonal_matrix)
        mvn_dist = torch.distributions.MultivariateNormal(mean2, cov2)
        beta_samples_taus = mvn_dist.sample(torch.Size([1000]))
        beta_samples_taus = beta_samples_taus.mean(dim=0)
        beta_for_tau.append(beta_samples_taus)

    tau_beta_samples = torch.stack(beta_for_tau)
    print("Betas from taus ", tau_beta_samples.mean(dim=0))

    beta_samples, beta_log_prob = beta_flows.sample_and_log_prob(num_samples=1000, context=context)
    beta_samples = beta_samples[0]
    print("Betas from standard flows : ", beta_samples.mean(dim=0))

    beta_samples = beta_samples.detach().cpu().numpy()
    tau_beta_samples = tau_beta_samples.detach().cpu().numpy()
    title = 'Violin Plot of Betas from Taus and Standard Flows'
    View.plot_betas_from_flows_vs_from_tau_flows(dimension, beta_samples, tau_beta_samples, title)


def experiment_compare_beta_path_from_taus_with_beta_path_group_lasso_for_lambda_range(dimension, X_train, Y_train,
                                                                                       likelihood_sigma, beta_flows,
                                                                                       tau_flows,
                                                                                       grouped_indices_list,
                                                                                       lambda_min_exp,
                                                                                       lambda_max_exp, context_size,
                                                                                       num_samples,
                                                                                       text=""):
    from torch.cuda.amp import autocast
    with torch.no_grad():
        with autocast():
            lambdas, tau_beta_samples_list, beta_samples_list = [], [], []
            for iter in range(200):
                print(iter)
                uniform_lambdas = torch.rand(context_size).to(device)
                context = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
                tau_samples, tau_log_prob = tau_flows.sample_and_log_prob(num_samples=num_samples, context=context)

                max_index = max(max(sublist) for sublist in grouped_indices_list) + 1
                repeat_counts = torch.tensor([len(sublist) for sublist in grouped_indices_list]).to(device)
                values_repeated = tau_samples.repeat_interleave(repeat_counts, dim=-1)
                flat_indices = [idx for sublist in grouped_indices_list for idx in sublist]
                flat_indices_tensor = torch.tensor(flat_indices).to(device)
                batch_size, depth, _ = tau_samples.shape
                list_tensors = torch.zeros((batch_size, depth, max_index)).to(device)
                list_tensors[:, :, flat_indices_tensor] = values_repeated.to(device)

                # batch_size = 10  # Adjust batch size as needed
                # diagonal_matrices = []
                # for i in range(0, len(list_tensors), batch_size):
                #     batch = list_tensors[i:i + batch_size]
                #     diagonal_batch = torch.diag_embed(batch)
                #     diagonal_matrices.append(diagonal_batch)
                # taus_diagonal_matrices_context = torch.cat(diagonal_matrices, dim=0)

                # taus_diagonal_matrices = []
                # batch_size_outer = 1
                # batch_size_inner = 1
                # for i in range(0, len(list_tensors), batch_size_outer):
                #     context_diagonal_matrix = None
                #     # diagonal_matrices = []
                #     batch_outer = list_tensors[i:i + batch_size_outer]
                #     for j in range(0, batch_outer.shape[1], batch_size_inner):
                #         print("At outer inner batch ", i, j)
                #         batch_inner = batch_outer[:, j:j + batch_size_inner, :]
                #         # print(batch_outer.shape, batch_inner.shape)
                #         diagonal_batch = torch.diag_embed(batch_inner)
                #         # diagonal_matrices.append(diagonal_batch)
                #         if context_diagonal_matrix is None:
                #             context_diagonal_matrix = diagonal_batch
                #         else:
                #             context_diagonal_matrix = torch.cat((context_diagonal_matrix, diagonal_batch), dim=0)
                #
                #     taus_diagonal_matrices.append(context_diagonal_matrix)
                # taus_diagonal_matrices_context = torch.cat(taus_diagonal_matrices, dim=0)

                taus_diagonal_matrices_context = generate_tau_diagonal_matrices(tau_samples, grouped_indices_list)

                mean_beta_dist, cov_beta_dist = calculate_beta_distribution_by_plugging_in_taus_in_main_equation(likelihood_sigma,
                                                                                               X_train, Y_train,
                                                                                               taus_diagonal_matrices_context)
                tau_beta_samples = mean_beta_dist
                beta_samples, beta_log_prob = beta_flows.sample_and_log_prob(num_samples=num_samples, context=context)

                lambdas.append((10 ** context).squeeze().cpu().detach().numpy())
                tau_beta_samples_list.append(tau_beta_samples.cpu().detach().numpy())
                beta_samples_list.append(beta_samples.cpu().detach().numpy())

            lambdas, tau_beta_samples_list, beta_samples_list = (np.concatenate(lambdas, 0),
                                                                     np.concatenate(tau_beta_samples_list, 0),
                                                                     np.concatenate(beta_samples_list, 0))
            lambda_sort_order = lambdas.argsort()
            lambdas_sorted = lambdas[lambda_sort_order]
            tau_beta_samples_sorted = tau_beta_samples_list[lambda_sort_order]
            beta_samples_sorted = beta_samples_list[lambda_sort_order]

            list_samples_sorted = [tau_beta_samples_sorted, beta_samples_sorted]
            title = "Standardized Coefficients - Betas-from-Taus Vs Betas-from-flows Vs Ground Truth -"+text
            View.plot_recovered_betas_vs_ground_truth_standardized_coefficients(X_train.detach().cpu().numpy(),
                                                                                Y_train.detach().cpu().numpy(),
                                                                                grouped_indices_list, lambdas_sorted,
                                                                                list_samples_sorted, title)

def main():
    # Set the parameters
    epochs = 500
    dimension = 300
    group_size = 30
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    zero_weight_group_index = 2
    data_sample_size = 50
    data_noise_sigma = 2.0
    likelihood_sigma = 2
    tau_sample_size = 1
    context_size = 1000
    lambda_min_exp = -2
    lambda_max_exp = 4
    learning_rate = 1e-3
    plot_sample_context_size = 10      # Increase to 1000 for lower dimensions
    plot_num_samples = 100               # Increase to 100 for lower dimensions

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, zero_weight_group_index:{zero_weight_group_index}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    # X, Y, W, variance = generate_synthetic_data(dimension, grouped_indices_list, zero_weight_group_index, data_sample_size, data_noise_sigma)
    # X, Y, W = generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list, data_sample_size,
    #                                                                data_noise_sigma)
    X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    X /= X.std(0)

    X_torch = X.to(device)
    Y_torch = Y.to(device)

    tau_flows, lambda_max_likelihood = posterior(X, Y, X_torch, Y_torch, likelihood_sigma, grouped_indices_list, epochs,
                                                 tau_sample_size, context_size,
                                                 lambda_min_exp, lambda_max_exp, learning_rate, W)

    beta_flows, lambda_max_likelihood_beta = GroupLassoRegressionCNF.posterior(X.detach().cpu().numpy(),
                                                                               Y.detach().cpu().numpy(),
                                                                               X_torch, Y_torch, likelihood_sigma,
                                                                               grouped_indices_list, epochs,
                                                                               tau_sample_size,
                                                                               context_size, lambda_min_exp,
                                                                               lambda_max_exp, learning_rate, W)

    # experiment_compare_betas_from_group_lasso_for_a_lambda(dimension, X_torch, Y_torch, likelihood_sigma,
    #                                           beta_flows, tau_flows, grouped_indices_list, lambda_max_likelihood)


    experiment_compare_beta_path_from_taus_with_beta_path_group_lasso_for_lambda_range(dimension, X_torch, Y_torch,
                                                                                       likelihood_sigma, beta_flows,
                                                                                       tau_flows,
                                                                                       grouped_indices_list,
                                                                                       lambda_min_exp,
                                                                                       lambda_max_exp,
                                                                                       plot_sample_context_size,
                                                                                       plot_num_samples)


if __name__ == "__main__":
    main()

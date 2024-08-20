import math

import numpy as np
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import ConditionalSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from torch import optim

import GroupPoissonRegressionCNF
import Utilities
import Visualizations as View

from enflows.transforms.nonlinearities import Softplus, Exp

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def log_posterior_unnormalized(grouped_indices_list, tau_samples, X, Z, lambdas_exp, sigma):
    lambdas = 10 ** lambdas_exp
    variance = sigma ** 2

    log_posterior = 0

    d = X.shape[-1]
    G = len(grouped_indices_list)
    log_posterior = log_posterior + ((d + G) / 2) * torch.log((lambdas ** 2) / 2)
    log_posterior = log_posterior + -(d / 2) * torch.log(2 * torch.tensor(torch.pi))

    term_0 = 0
    for group_indices in grouped_indices_list:
        term_0 = term_0 - torch.lgamma(0.5 * torch.tensor(len(group_indices) + 1))
    log_posterior = log_posterior + term_0

    max_index = max(max(sublist) for sublist in grouped_indices_list) + 1
    repeat_counts = torch.tensor([len(sublist) for sublist in grouped_indices_list]).to(device)
    values_repeated = tau_samples.repeat_interleave(repeat_counts, dim=-1)
    flat_indices = [idx for sublist in grouped_indices_list for idx in sublist]
    flat_indices_tensor = torch.tensor(flat_indices).to(device)
    batch_size, depth, _ = tau_samples.shape
    list_tensors = torch.zeros((batch_size, depth, max_index)).to(device)
    list_tensors[:, :, flat_indices_tensor] = values_repeated.to(device)
    taus_diagonal_matrices = torch.diag_embed(1 / list_tensors)
    term_3 = (1 / variance) * torch.matmul(X.T, X) + taus_diagonal_matrices

    log_posterior = log_posterior + -0.5 * torch.sum(torch.log(tau_samples), dim=-1).unsqueeze(-1)
    log_posterior = log_posterior + ((- lambdas ** 2 / 2) * tau_samples.sum(dim=-1)).unsqueeze(-1)
    log_posterior = log_posterior + -0.5 * torch.linalg.slogdet(term_3)[1].unsqueeze(-1)
    log_posterior = log_posterior.squeeze(-1)
    return log_posterior


def train_CNF(flows, d, grouped_indices_list, X, Z, X_torch, Z_torch, likelihood_sigma, epochs, flow_sample_size,
              context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
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
        for epoch in range(epochs):
            t = epoch // (epochs / cool_num_iter)
            T = cooling_function(t=t)

            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = lambdas_exp
            tau_samples, flow_log_prob = flows.sample_and_log_prob(num_samples=flow_sample_size, context=context)
            log_p = log_posterior_unnormalized(grouped_indices_list, tau_samples, X_torch, Z_torch,
                                               lambdas_exp, likelihood_sigma)
            log_p = torch.clamp(log_p, min=-1e10, max=1e10)

            loss = torch.mean(flow_log_prob - (log_p / T))
            # loss = torch.mean(flow_log_prob - log_p)
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
                lambdas_sorted, tau_samples_sorted, losses_sorted = sample_from_flow_for_plots(
                    flows,
                    grouped_indices_list,
                    X_torch, Z_torch,
                    likelihood_sigma, 100, 100,
                    lambda_min_exp,
                    lambda_max_exp)
                log_likelihood_means = np.mean(-losses_sorted, axis=1)
                lambda_max_likelihood = lambdas_sorted[np.argmax(log_likelihood_means)]

                solution_type = "No_Betas_Poisson-Group-Lasso-Solution Path"
                View.plot_flow_group_coefficients_path_vs_ground_truth(X, Z, lambdas_sorted, tau_samples_sorted,
                                                                       solution_type)

                # title = "GL-Without_betas_Log_marginal_likelihood"
                # View.plot_log_marginal_likelihood_vs_lambda(X, Y, lambdas_sorted, losses_sorted, likelihood_sigma ** 2,
                #                                             title)

    except KeyboardInterrupt:
        print("interrupted..")

    return flows, losses, lambda_max_likelihood


def generate_synthetic_data(d, n, noise):
    # Define a Posisson distribution and generate some real world samples X and Y
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))

    data_mean = torch.zeros(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.randn(d)

    min_val = torch.min(W)
    max_val = torch.max(W)
    W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W)

    v = torch.tensor(noise ** 2)
    delta = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + delta
    mean_poisson = torch.exp(Y)
    Z = torch.poisson(mean_poisson) + 1

    return X, Z, W, v, Y, mean_poisson


def build_sum_of_sigmoid_conditional_flow_model(d):
    context_features = 16
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 3
    for _ in range(num_layers):
        transforms.append(
            InverseTransform(
                ConditionalSumOfSigmoidsTransform(
                    features=d, hidden_features=64,
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
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=64,
                                num_blocks=3, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def sample_from_flow_for_plots(flows, grouped_indices_list, X, Z, likelihood_sigma, context_size, flow_sample_size,
                               lambda_min_exp, lambda_max_exp):
    num_iter = 10
    lambdas, tau_samples_list, losses = [], [], []

    with torch.no_grad():
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            tau_samples, flow_log_prob = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_p_samples = log_posterior_unnormalized(grouped_indices_list, tau_samples, X, Z,
                                                       lambdas_exp, likelihood_sigma)

            loss = flow_log_prob - log_p_samples

            lambdas.append((10 ** lambdas_exp).squeeze().cpu().detach().numpy())
            tau_samples_list.append(tau_samples.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

    tau_samples_list, lambdas, losses = (np.concatenate(tau_samples_list, 0),
        np.concatenate(lambdas, 0), np.concatenate(losses, 0))
    lambda_sort_order = lambdas.argsort()

    lambdas_sorted = lambdas[lambda_sort_order]
    tau_samples_sorted = tau_samples_list[lambda_sort_order]
    losses_sorted = losses[lambda_sort_order]
    return lambdas_sorted, tau_samples_sorted, losses_sorted


def posterior(X, Z, X_torch, Z_torch, likelihood_sigma, grouped_indices_list, epochs, tau_sample_size,
              context_size, lambda_min_exp, lambda_max_exp, learning_rate, W):
    n_groups = len(grouped_indices_list)
    dimension = n_groups

    # ==================================================================
    # train conditional flows

    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows.to(device)

    flows, losses, lambda_max_likelihood = train_CNF(flows, dimension, grouped_indices_list, X, Z, X_torch, Z_torch,
                                                     likelihood_sigma, epochs,
                                                     tau_sample_size,
                                                     context_size, lambda_min_exp, lambda_max_exp,
                                                     learning_rate)
    print("Best lamba selected from flows : ", lambda_max_likelihood)

    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    View.plot_loss(losses)
    # solution_type = "No_Beta_Group-Lasso-Solution Path"
    solution_type = "No_Beta_Group-Poisson-Lasso-MAP"
    lambdas_sorted, tau_samples_sorted, losses_sorted = sample_from_flow_for_plots(flows, grouped_indices_list, X_torch,
                                                                                   Z_torch, likelihood_sigma,
                                                                                   100, 100,
                                                                                    lambda_min_exp, lambda_max_exp)

    View.plot_flow_group_coefficients_path_vs_ground_truth(X, Z, lambdas_sorted, tau_samples_sorted, solution_type)
    return flows, lambda_max_likelihood


def generate_group_indices(grouped_indices_list):
    total_indices = len(set(index for sublist in grouped_indices_list for index in sublist))

    group_indices = [0] * total_indices

    for group_id, indices in enumerate(grouped_indices_list):
        for index in indices:
            group_indices[index] = group_id

    return group_indices


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


def main():
    # Set the parameters
    epochs = 1000
    dimension = 12
    group_size = 3
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    zero_weight_group_index = 2
    data_sample_size = 100
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
    X, Z, W, variance, Y, mean_poisson = generate_synthetic_data_with_zero_group_coefficients(dimension,
                                                                                              grouped_indices_list,
                                                                                              data_sample_size,
                                                                                              data_noise_sigma)
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data(dimension, data_sample_size, data_noise_sigma)

    # X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    X /= X.std(0)

    train_ratio = 0.8
    X_train, Z_train, X_test, Z_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Z)

    X_torch = X_train.to(device)
    Z_torch = Z_train.to(device)
    X_test, Z_test = X_test.to(device), Z_test.to(device)

    flows, lambda_max_likelihood = posterior(X_train, Z_train, X_torch, Z_torch, likelihood_sigma, grouped_indices_list, epochs,
                                  flow_sample_size, context_size,
                                  lambda_min_exp, lambda_max_exp, learning_rate, W)
    q_selected = Utilities.select_q_for_max_likelihood_lambda(lambda_max_likelihood, flows, device)

    print(q_selected)


if __name__ == "__main__":
    main()

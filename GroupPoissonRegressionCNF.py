import math

import numpy as np
import scipy as sp
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import ConditionalSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from sklearn.linear_model import LinearRegression

from torch import optim

import Evaluation
import Utilities
import Visualizations as View

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
from rpy2.robjects import pandas2ri

pandas2ri.activate()
torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def vectorized_log_likelihood_unnormalized(Ws, X, Z, context, likelihood_sigma):
    n = Z.shape[0]
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    etas = XWs.squeeze()
    term_1 = torch.matmul(etas, Z)
    term_2 = -1 * torch.sum(torch.exp(etas), dim=-1)
    term_3 = torch.sum(torch.lgamma(Z + 1))  # For an integer n, log(n!) = log(Γ(n+1))
    log_likelihood = term_1 + term_2 - term_3
    return log_likelihood


def sum_of_norms_of_W_groups(Ws, indices_list):
    # sum_tensor = torch.zeros(Ws.shape[: -1]).to(device)
    # for indices in indices_list:
    #     sum_tensor = sum_tensor + torch.norm(Ws[:, :, indices], p=2, dim=-1)
    indices_tensor = torch.tensor(indices_list, dtype=torch.long)
    Ws_gathered = Ws[:, :, indices_tensor]
    norms = torch.norm(Ws_gathered, p=2, dim=-1)
    sum_tensor = norms.sum(dim=-1).to(device)
    return sum_tensor


def vectorized_log_prior_unnormalized(Ws, d, grouped_indices_list, context, std_dev):
    lambdas_exp = context
    lambdas_list = (10 ** lambdas_exp)
    G = len(grouped_indices_list)
    sum_tensor = sum_of_norms_of_W_groups(Ws, grouped_indices_list)
    term_1 = G * torch.log(lambdas_list / std_dev)
    term_2 = -(lambdas_list / std_dev) * sum_tensor
    log_prior = term_1 + term_2
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, grouped_indices_list, X, Z, context, likelihood_sigma):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Z, context, likelihood_sigma)
    log_prior = vectorized_log_prior_unnormalized(q_samples, d, grouped_indices_list, context, likelihood_sigma)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train_CNF(flows, d, grouped_indices_list, X, Z, X_torch, Z_torch, likelihood_sigma, epochs, n, context_size=100,
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
            q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=n, context=context)

            log_p = vectorized_log_posterior_unnormalized(q_samples, d, grouped_indices_list, X_torch, Z_torch,
                                                          context, likelihood_sigma)
            loss = torch.mean(q_log_prob - (log_p / T))
            # loss = torch.mean(q_log_prob - log_p)

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
                solution_type = "Poisson-Group-Lasso-Solution Path"
                lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Z_torch,
                                                                                      likelihood_sigma,
                                                                                      grouped_indices_list, 100,
                                                                                      100,
                                                                                      lambda_min_exp, lambda_max_exp)

                log_likelihood_means = np.mean(-losses_sorted, axis=1)
                lambda_max_likelihood = lambdas_sorted[np.argmax(log_likelihood_means)]

                View.plot_flow_group_poisson_path_vs_ground_truth(X, Z, grouped_indices_list,
                                                                  lambdas_sorted, q_samples_sorted, solution_type)
                # title = "Group-Lasso-Regression-CNF"
                # View.plot_log_marginal_likelihood_vs_lambda(X, Z, lambdas_sorted, losses_sorted, likelihood_sigma ** 2,
                #                                             title, grouped_indices_list)

    except KeyboardInterrupt:
        print("interrupted..")

    return flows, losses, lambda_max_likelihood


def generate_synthetic_data(d, grouped_indices_list, zero_weight_group_index, n, noise):
    # Define a Posisson distribution and generate some real world samples X and Y
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))

    num_data_samples = torch.Size([n])
    num_samples = n
    X = torch.zeros((num_samples, d))

    mean = 1
    std = 0.1

    g1_size = len(grouped_indices_list[0])
    g1_mean = torch.normal(mean=mean, std=std, size=(num_samples, g1_size))
    X[:, grouped_indices_list[0]] = g1_mean + torch.normal(mean=0, std=0.1, size=(num_samples, g1_size))

    g2_size = len(grouped_indices_list[1])
    # g2_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    g2_base = torch.normal(mean=mean, std=std, size=(num_samples, g2_size))
    X[:, grouped_indices_list[1]] = g2_base + torch.normal(mean=0, std=0.1, size=(num_samples, g2_size))

    g3_size = len(grouped_indices_list[2])
    # g3_base = torch.rand(num_samples, 1)
    # X[:, grouped_indices_list[2]] = g3_base + torch.rand(num_samples, g3_size)
    g3_base = torch.normal(mean=mean, std=std, size=(num_samples, g3_size))
    X[:, grouped_indices_list[2]] = g3_base + torch.normal(mean=0, std=0.1, size=(num_samples, g3_size))

    # 15, 100
    for group_index in range(len(grouped_indices_list) - 3):
        g_size = len(grouped_indices_list[group_index + 3])
        g_base = torch.normal(mean=mean, std=std, size=(num_samples, g_size))
        X[:, grouped_indices_list[group_index + 3]] = g_base + torch.normal(mean=0, std=0.1, size=(num_samples, g_size))

    # W = torch.rand(d) * 20 - 10
    W = torch.randn(d)

    min_val = torch.min(W)
    max_val = torch.max(W)
    W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    # W[grouped_indices_list[zero_weight_group_index]] = 0

    v = torch.tensor(noise ** 2)
    delta = torch.randn(num_data_samples) * v
    # delta = torch.normal(0, noise ** 2, num_data_samples)
    Y = torch.matmul(X, W) + delta
    mean_poisson = torch.exp(Y)
    Z = torch.poisson(mean_poisson) + 1
    return X, Z, W, v, Y, mean_poisson


def generate_synthetic_data_2(d, n, noise):
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

    transforms = transforms[::-1]
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=64,
                                num_blocks=3, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def print_original_vs_flow_learnt_parameters(d, fixed, flows, context=None):
    sample_size = 10000
    sample_mean = None
    if context is None:
        q_samples = flows.sample(sample_size)
        sample_mean = torch.mean(q_samples, dim=0).tolist()
    else:
        context = context.to(device)
        q_samples = flows.sample(sample_size, context=context.view(-1, 1))
        sample_mean = torch.mean(q_samples[0], dim=0).tolist()
        print("For Context : ", context)

    print(f"Index ||  Original  ||  Fixed Lambda ")
    for i in range(d):
        print(f"Index {i}       :     {fixed[i]}      :       {sample_mean[i]}")


def sample_Ws_for_plots(flows, X, Z, likelihood_sigma, grouped_indices_list, context_size, flow_sample_size,
                        lambda_min_exp, lambda_max_exp):
    d = X.shape[1]
    num_iter = 10
    lambdas, q_samples_list, losses = [], [], []

    with torch.no_grad():
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            q_samples, q_log_probs = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_p_samples = vectorized_log_posterior_unnormalized(q_samples, d, grouped_indices_list, X, Z, lambdas_exp,
                                                                  likelihood_sigma)
            loss = q_log_probs - log_p_samples

            lambdas.append((10 ** lambdas_exp).squeeze().cpu().detach().numpy())
            q_samples_list.append(q_samples.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

    q_samples_list, lambdas, losses = (np.concatenate(q_samples_list, 0),
                                       np.concatenate(lambdas, 0), np.concatenate(losses, 0))
    lambda_sort_order = lambdas.argsort()

    lambdas_sorted = lambdas[lambda_sort_order]
    q_samples_sorted = q_samples_list[lambda_sort_order]
    losses_sorted = losses[lambda_sort_order]
    return lambdas_sorted, q_samples_sorted, losses_sorted


def posterior(X, Z, X_torch, Z_torch, likelihood_sigma, grouped_indices_list, epochs, q_sample_size,
              context_size, lambda_min_exp, lambda_max_exp, learning_rate, W):
    dimension = X_torch[0].shape[0]
    original_W = W.tolist()
    print("Original Parameters: ", original_W)

    # ==================================================================
    # train conditional flows

    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows.to(device)

    flows, losses, lambda_max_likelihood = train_CNF(flows, dimension, grouped_indices_list, X, Z, X_torch, Z_torch,
                              likelihood_sigma, epochs,
                              q_sample_size,
                              context_size, lambda_min_exp, lambda_max_exp,
                              learning_rate
                              )

    print("Best lamba selected from flows : ", lambda_max_likelihood)


    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    View.plot_loss(losses)
    solution_type = "Poisson-Group-Lasso-Solution Path"
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Z_torch,
                                                                          likelihood_sigma, grouped_indices_list, 100,
                                                                          100,
                                                                          lambda_min_exp, lambda_max_exp)
    solution_type = "Poisson-Group-Lasso-Solution Path - MAP"
    View.plot_flow_group_poisson_path_vs_ground_truth(X, Z, grouped_indices_list,
                                                      lambdas_sorted, q_samples_sorted, solution_type)
    #
    # View.plot_group_norms_vs_lambda(X, Z, grouped_indices_list, lambdas_sorted, q_samples_sorted)
    #
    # View.plot_flow_group_lasso_path_vs_ground_truth_standardized_coefficients(X, Z, grouped_indices_list,
    #                                                                           lambdas_sorted, q_samples_sorted,
    #                                                                           solution_type)

    return flows, lambda_max_likelihood


def main():
    # Set the parameters
    epochs = 5000
    dimension = 24
    group_size = 3
    grouped_indices_list = [list(range(i, i + group_size)) for i in range(0, dimension, group_size)]
    zero_weight_group_index = 1
    data_sample_size = 60
    data_noise_sigma = 1.0
    likelihood_sigma = 1
    q_sample_size = 1
    context_size = 1000
    lambda_min_exp = 0
    lambda_max_exp = 10
    learning_rate = 1e-3

    data_sample_size = 30
    data_noise_sigma = 1.0
    likelihood_sigma = 1
    q_sample_size = 1
    context_size = 1000
    lambda_min_exp = -2
    lambda_max_exp = 6
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, zero_weight_group_index:{zero_weight_group_index}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data(dimension, grouped_indices_list,
    #                                                              zero_weight_group_index,
    #                                                              data_sample_size, data_noise_sigma)
    # X, Z, W, variance, Y, mean_poisson = generate_synthetic_data_2(dimension, data_sample_size, data_noise_sigma)
    X, Z, W = generate_synthetic_data_with_zero_group_coefficients(dimension, grouped_indices_list, data_sample_size,
                                                                   data_noise_sigma)
    X = (X - X.mean(0)) / X.std(0)

    train_ratio = 0.8
    X_train, Z_train, X_test, Z_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Z)

    X_torch = X_train.to(device)
    Z_torch = Z_train.to(device)
    X_test, Z_test = X_test.to(device), Z_test.to(device)

    flows, lambda_max_likelihood = posterior(X_train.detach().cpu().numpy(), Z_train.detach().cpu().numpy(), X_torch, Z_torch, likelihood_sigma,
              grouped_indices_list, epochs, q_sample_size, context_size,
              lambda_min_exp, lambda_max_exp, learning_rate, W)
    q_selected = Utilities.select_q_for_max_likelihood_lambda(lambda_max_likelihood, flows, device)

    print(q_selected)
    # Evaluation.evaluate_poisson_model(flows, q_selected, X_torch, Z_torch, "Poisson-Group-Lasso-Regression-CNF-Training-data")
    # Evaluation.evaluate_poisson_model(flows, q_selected, X_test, Z_test, "Poisson-Group-Lasso-Regression-CNF-Test-data")

    # PoissonRegressionCNF.posterior(X.detach().cpu().numpy(), Z.detach().cpu().numpy(), X_torch, Z_torch,
    #                                likelihood_sigma,
    #                                epochs, q_sample_size, context_size,
    #                                lambda_min_exp, lambda_max_exp, learning_rate, W, title="Lasso")


if __name__ == "__main__":
    main()

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

from torch import optim

import Evaluation
import Utilities
import Visualizations as View

from rpy2.robjects import pandas2ri

pandas2ri.activate()
torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def vectorized_log_likelihood_unnormalized(Ws, X, Z):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    etas = XWs.squeeze()
    term_1 = torch.matmul(etas, Z)
    term_2 = -1 * torch.sum(torch.exp(etas), dim=-1)
    term_3 = torch.sum(torch.lgamma(Z + 1))  # For an integer n, log(n!) = log(Γ(n+1))
    log_likelihood = term_1 + term_2 - term_3
    return log_likelihood


def vectorized_log_ridge_prior_unnormalized(Ws, sigma, context, d):
    lambdas_exp = context
    variance = torch.tensor(sigma) ** 2
    lambdas_list = 10 ** lambdas_exp
    squared_weights = (Ws * Ws).sum(dim=2)
    term_1 = -0.5 * d * (torch.log(2 * torch.pi * variance) - torch.log(lambdas_list))
    term_2 = lambdas_list * (-0.5 * (1 / variance) * squared_weights)
    log_prior = term_1 + term_2
    return log_prior


def vectorized_standard_laplace_log_prior_unnormalized(Ws, d, lambdas_exp):
    lambdas_list = (10 ** lambdas_exp)
    term_1 = d * torch.log(lambdas_list / 2.0)
    term_2 = -lambdas_list * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    return log_prior


def train_CNF(flows, d, X, Z, X_torch, Z_torch, likelihood_sigma, epochs, n, context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3, is_ridge_posterior=True):
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

            log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X_torch, Z_torch)
            if is_ridge_posterior:
                log_prior = vectorized_log_ridge_prior_unnormalized(q_samples, likelihood_sigma, context, d)
            else:
                log_prior = vectorized_standard_laplace_log_prior_unnormalized(q_samples, d, context)
            log_posterior = log_likelihood + log_prior

            loss = torch.mean(q_log_prob - (log_posterior / T))
            # loss = torch.mean(q_log_prob - log_posterior)

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
                solution_type = "Ridge Solution Path" if is_ridge_posterior else "Lasso Solution Path"

                lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Z_torch,
                                                                                      likelihood_sigma, 100,
                                                                                      100,
                                                                                      lambda_min_exp, lambda_max_exp)

                log_likelihood_means = np.mean(-losses_sorted, axis=1)
                lambda_max_likelihood = lambdas_sorted[np.argmax(log_likelihood_means)]

                View.plot_flow_poisson_regression_path_vs_ground_truth(X_torch.cpu().detach().numpy(),
                                                                       Z_torch.cpu().detach().numpy(),
                                                                       lambdas_sorted, q_samples_sorted, likelihood_sigma,
                                                                       solution_type, is_ridge_posterior)

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


def sample_Ws_for_plots(flows, X, Z, likelihood_sigma, context_size, flow_sample_size,
                        lambda_min_exp, lambda_max_exp, ridge=True):
    d = X.shape[1]
    num_iter = 10
    lambdas, q_samples_list, losses = [], [], []

    with torch.no_grad():
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            q_samples, q_log_probs = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Z)
            if ridge:
                log_prior = vectorized_log_ridge_prior_unnormalized(q_samples, likelihood_sigma, lambdas_exp, d)
            else:
                log_prior = vectorized_standard_laplace_log_prior_unnormalized(q_samples, d, lambdas_exp)
            log_posterior = log_likelihood + log_prior

            loss = q_log_probs - log_posterior

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


def posterior(X, Z, X_torch, Z_torch, likelihood_sigma, epochs, q_sample_size,
              context_size, lambda_min_exp, lambda_max_exp, learning_rate, W, title):
    dimension = X_torch[0].shape[0]
    original_W = W.tolist()
    print("Original Parameters: ", original_W)

    if title.lower() == "ridge":
        is_ridge_posterior = True
    elif title.lower() == "lasso":
        is_ridge_posterior = False
    else:
        raise ValueError("Title must be either ridge or lasso!")

    print("======================= Training Poisson Ridge Regression =======================")
    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows.to(device)
    flows, losses, lambda_max_likelihood = train_CNF(flows, dimension, X, Z, X_torch, Z_torch,
                                                     likelihood_sigma, epochs,
                                                     q_sample_size,
                                                     context_size, lambda_min_exp, lambda_max_exp,
                                                     learning_rate, is_ridge_posterior=is_ridge_posterior)

    print("Best lamba selected from flows : ", lambda_max_likelihood)

    # View.plot_loss(losses)
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Z_torch,
                                                                          likelihood_sigma, 100,
                                                                          100,
                                                                          lambda_min_exp,
                                                                          lambda_max_exp,
                                                                          ridge=is_ridge_posterior)

    solution_type = "Poisson-" + title + "-Solution Path - MAP"

    View.plot_flow_poisson_regression_path_vs_ground_truth(X_torch.cpu().detach().numpy(),
                                                           Z_torch.cpu().detach().numpy(),
                                                           lambdas_sorted, q_samples_sorted, likelihood_sigma,
                                                           solution_type, is_ridge_posterior)

    return flows, lambda_max_likelihood


def main():
    # Set the parameters
    epochs = 1000
    dimension = 8
    data_sample_size = 50
    data_noise_sigma = 1.0
    likelihood_sigma = 1
    q_sample_size = 1
    context_size = 100
    lambda_min_exp = -2
    lambda_max_exp = 6
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension},"
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    X, Z, W, variance, Y, mean_poisson = generate_synthetic_data(dimension, data_sample_size, data_noise_sigma)

    X = (X - X.mean(0)) / X.std(0)

    train_ratio = 0.8
    X_train, Z_train, X_test, Z_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Z)


    X_torch = X_train.to(device)
    Z_torch = Z_train.to(device)
    X_test, Z_test = X_test.to(device), Z_test.to(device)

    print("======================= Poisson Ridge Regression =======================")
    flows, lambda_max_likelihood = posterior(X_train.detach().cpu().numpy(), Z_train.detach().cpu().numpy(), X_torch, Z_torch, likelihood_sigma,
              epochs, q_sample_size, context_size,
              lambda_min_exp, lambda_max_exp, learning_rate, W, title="Ridge")
    q_selected = Utilities.select_q_for_max_likelihood_lambda(lambda_max_likelihood, flows, device)

    Evaluation.evaluate_poisson_model(flows, q_selected, X_torch, Z_torch, "Poisson-Ridge-Regression-CNF-Training-data")
    Evaluation.evaluate_poisson_model(flows, q_selected, X_test, Z_test, "Poisson-Ridge-Regression-CNF-Test-data")

    print("=======================  Poisson Lasso Regression =======================")
    flows, lambda_max_likelihood = posterior(X_train.detach().cpu().numpy(), Z_train.detach().cpu().numpy(), X_torch, Z_torch, likelihood_sigma,
              epochs, q_sample_size, context_size,
              lambda_min_exp, lambda_max_exp, learning_rate, W, title="Lasso")

    Evaluation.evaluate_poisson_model(flows, q_selected, X_torch, Z_torch, "Poisson-Lasso-Regression-CNF-Training-data")
    Evaluation.evaluate_poisson_model(flows, q_selected, X_test, Z_test, "Poisson-Lasso-Regression-CNF-Test-data")


if __name__ == "__main__":
    main()

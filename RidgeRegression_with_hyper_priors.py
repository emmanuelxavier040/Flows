"""
This is an example of posterior estimation using flows by minimizing the KL divergence with synthetic data.
We use a Multi-variate normal distribution to generate X. We choose a fixed parameter W which can be used for a linear
transformation of X to Y. We can add some noise to the observations finally giving Y = XW + noise. Our task is to infer
about the posterior P(W | X,Y). We use linear flows to compute the KL-Divergence(q(W) || P*(W | X,Y)). Here P* is the
un-normalized posterior which is equivalent to the un-normalized Gaussian likelihood * Gaussian prior.
We can compute P* since I have X, Y and W (for W, we can easily sample from flow). After training, flows should have
learned the distribution of W and samples from it should resemble the fixed W which we used to transform X to Y.
"""

import torch
from enflows.transforms import ActNorm
from scipy.stats import multivariate_normal
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift


import Visualizations as View
import Utilities

torch.manual_seed(11)

def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=-1)
    return log_likelihood


def vectorized_log_likelihood_t_distribution_unnormalized(Ws, d, X, Y):
    a_0 = 2 * len(X)
    b_0 = a_0
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    term_1 = -(a_0 + d / 2)
    term_2 = torch.log(1 + (1 / (2 * b_0)) * torch.sum(squared_errors, dim=-1))
    log_likelihood = term_1 * term_2
    return log_likelihood


def vectorized_log_prior_unnormalized(Ws, variance):
    outputs = torch.matmul(Ws.unsqueeze(1), Ws.unsqueeze(-1))
    squared_weights = outputs.squeeze()
    log_prior = -0.5 * (1 / variance) * squared_weights
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    # log_likelihood =  vectorized_log_likelihood_t_distribution_unnormalized(q_samples, d, X, Y)
    log_prior = vectorized_log_prior_unnormalized(q_samples, variance)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def vectorized_log_posterior_predictive_unnormalized(y_samples, y_length, X_train, y_train, X_test, variance):
    beta_dimension = X_train[0].shape[0]
    μ_0 = torch.zeros(beta_dimension)
    cov_0 = torch.eye(beta_dimension)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X_train.t(), X_train) + Λ_0
    Λ_N_1 = Utilities.woodbury_matrix_conversion(Λ_0, X_train.t(), torch.eye(X_train.shape[0]), X_train, device="cpu")
    μ_N = torch.matmul(Λ_N_1, (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X_train.t(), y_train)))
    mean = torch.matmul(X_test, μ_N)
    # cov = variance * (I +  X_test * Λ_N * X_test.t())
    cov_1 = Utilities.woodbury_matrix_conversion(variance * torch.eye(y_length), variance*X_test, Λ_N, X_test.t(), device="cpu")
    term_1 = y_samples - mean
    log_posterior_predictive = -0.5 * torch.bmm(term_1.unsqueeze(1),
              torch.matmul(cov_1, term_1.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)).squeeze()
    return log_posterior_predictive


def train_posterior(flows, d, X, Y, variance, epoch, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    losses = []
    for i in range(epoch):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, variance)
        loss = torch.mean(q_log_prob - log_p)
        if i == 0 or i % 100 == 0 or i + 1 == epoch:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def train_posterior_predictive(flows, d, X, Y, X_test, variance, epoch, y_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    losses = []
    for i in range(epoch):
        optimizer.zero_grad()
        y_samples, y_log_prob = flows.sample_and_log_prob(y_sample_size)
        log_p = vectorized_log_posterior_predictive_unnormalized(y_samples, d, X, Y, X_test, variance)
        loss = torch.mean(y_log_prob - log_p)
        if i == 0 or i % 100 == 0 or i + 1 == epoch:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def train_with_student_t(flows, d, X, Y, variance, epoch, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows with Student-t likelihood")
    for i in range(epoch):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_likelihood = vectorized_log_likelihood_t_distribution_unnormalized(q_samples, d, X, Y)
        log_prior = vectorized_log_prior_unnormalized(q_samples, variance)
        log_p = log_likelihood + log_prior
        # print(log_likelihood, "============")
        # print(log_prior)
        loss = torch.mean(q_log_prob - log_p)
        if i == 0 or i % 100 == 0 or i + 1 == epoch:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        loss.backward()
        optimizer.step()
    return flows


def sample_Ws(model, sample_size, X, Y, min_lambda, max_lambda, variance, interval_size, n_iter):
    sample_list, lambda_list, loss_list = [], [], []
    with torch.no_grad():
        for _ in range(n_iter):
            uniform_lambdas = torch.rand(interval_size).cuda()
            lambdas_exp = (uniform_lambdas * (max_lambda - min_lambda) + min_lambda).view(-1, 1)
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)
            posterior_eval = vectorized_log_posterior_unnormalized(posterior_samples, X, Y, variance)


def generate_synthetic_data(d, n, n_test):
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    v = torch.tensor(0.5)
    noise = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + noise

    num_test_samples = torch.Size([n_test])
    X_test = data_mvn_dist.sample(num_test_samples)
    noise = torch.randn(n_test) * v
    Y_test = torch.matmul(X_test, W) + noise

    return X, Y, W, v, X_test, Y_test


def build_flow_model(d):
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 15
    for _ in range(num_layers):
        transforms.append(InverseTransform(NaiveLinear(features=d)))
        transforms.append(InverseTransform(ScalarScale(scale=1.5)))
        transforms.append(InverseTransform(ScalarShift(shift=0.5)))
        transforms.append(InverseTransform(ActNorm(features=d)))

    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


def compute_analytical_posterior_for_fixed_variance(X, Y, prior_mean, prior_covariance, sigma_2):
    # Gaussian distribution P(Beta | X, Y, sigma_2)
    X_transpose_X = torch.matmul(X.t(), X)
    inv_prior_covariance = torch.inverse(prior_covariance)
    inv_posterior_covariance = inv_prior_covariance + X_transpose_X
    posterior_covariance = torch.inverse(inv_posterior_covariance)
    term_2 = torch.matmul(prior_mean.t(), inv_prior_covariance) + torch.matmul(X.t(), Y)
    posterior_mean = torch.matmul(posterior_covariance, term_2)
    mean_np = posterior_mean.numpy()
    cov_np = sigma_2 * posterior_covariance.numpy()
    return mean_np, cov_np


def compute_posterior_predictive_normal_dist_for_fixed_variance(X_train, y_train, variance, X_pred):
    beta_dimension = X_train[0].shape[0]
    μ_0 = torch.zeros(beta_dimension)
    cov_0 = torch.eye(beta_dimension)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X_train.t(), X_train) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X_train.t(), y_train)))

    mean = torch.matmul(X_pred, μ_N)
    covariance = variance * (torch.eye(len(X_pred)) + torch.matmul(torch.matmul(X_pred, Λ_N), X_pred.t()))
    return mean, covariance


def posterior(X_train, Y_train, W, variance):
    num_iter = 2000
    q_sample_size = 100
    dimension = X_train[0].shape[0]
    flows = build_flow_model(dimension)
    flows, losses = train_posterior(flows, dimension, X_train, Y_train, variance, num_iter, q_sample_size)
    # View.plot_loss(losses)

    analytical_mean, analytical_cov = compute_analytical_posterior_for_fixed_variance(X_train, Y_train,
                                                                                      torch.zeros(dimension),
                                                                                      torch.eye(dimension), variance)
    View.plot_analytical_flow_posterior_general_with_samples(analytical_mean, analytical_cov, flows)
    View.plot_analytical_flow_distribution_on_grid(dimension, analytical_mean, analytical_cov, flows,
                                                   W, "Normal-Posterior-General")
    q_samples = flows.sample(10000)
    sample_mean_1 = torch.mean(q_samples, dim=0).tolist()
    fixed = W.tolist()
    q_samples = flows.sample(10000)
    sample_mean = torch.mean(q_samples, dim=0).tolist()
    for i in range(dimension):
        print(f"Index {i}: {fixed[i]} Fixed Sigma W: {sample_mean_1[i]} Student-t W: {sample_mean[i]}")

    return flows


def posterior_predictive(X_train, Y_train, X_test, Y_test):
    num_iter = 2000
    y_sample_size = 100
    num_test = Y_test.shape[0]
    variance = 4
    print("Post. pred. Dimension: ", num_test)
    flows = build_flow_model(num_test)
    flows, losses = train_posterior_predictive(flows, num_test, X_train, Y_train, X_test, variance, num_iter,
                                               y_sample_size)
    # View.plot_loss(losses)

    mean_pred, cov_pred = compute_posterior_predictive_normal_dist_for_fixed_variance(X_train, Y_train, variance, X_test)
    View.plot_analytical_flow_distribution_on_grid(num_test, mean_pred, cov_pred, flows,
                                                   Y_test, "Normal-Posterior-Predictive-General")
    print("Real Response: ", Y_test)
    print("Analytical Posterior Predictive Mean: ", mean_pred)

    return flows


def main():
    # Define a Multivariate-Normal distribution and generate some real world samples X
    dimension = 3
    num_data_samples = 100
    num_test_samples = 3
    X_train, Y_train, W, variance, X_test, Y_test = generate_synthetic_data(dimension, num_data_samples, num_test_samples)

    # Train posterior flows
    # posterior(X_train, Y_train, W, variance)

    # Train posterior predictive flows
    posterior_predictive(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()

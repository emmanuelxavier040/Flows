"""
This is an example of posterior estimation using flows by minimizing the KL divergence with synthetic data.
We use a Multi-variate normal distribution to generate X. We choose a fixed parameter W which can be used for a linear
transformation of X to Y. We can add some noise to the observations finally giving Y = XW + noise. Our task is to infer
about the posterior P(W | X,Y). We use linear flows to compute the KL-Divergence(q(W) || P*(W | X,Y)). Here P* is the
un-normalized posterior which is equivalent to the un-normalized Gaussian likelihood * Gaussian prior.
We can compute P* since I have X, Y and W (for W, we can easily sample from flow). After training, flows should have
learned the distribution of W and samples from it should resemble the fixed W which we used to transform X to Y.
"""
import numpy as np
import math
import torch
from enflows.transforms import ActNorm
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift
from scipy.stats import multivariate_normal, multivariate_t

torch.manual_seed(11)
np.random.seed(10)

import Visualizations as View


def vectorized_log_t_distribution_unnormalized(Ws, d, X, Y):
    N = len(X)
    μ_0 = torch.zeros(d)
    cov_0 = torch.eye(d)
    a_0 = torch.tensor(N)
    b_0 = a_0
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), Y)))
    a_N = a_0 + (d / 2.)
    b_N = b_0 + 0.5 * (torch.matmul(Y.t(), Y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_N.t(), Λ_N), μ_N))

    term_1 = -(a_N + d/2)
    term_4 = Ws - μ_N
    term_3 = torch.matmul(torch.matmul(term_4.unsqueeze(1), (a_N/b_N)*Λ_N), term_4.unsqueeze(-1)).squeeze(1)
    log_t_unnormalized = term_1 * torch.log(1 + (1/(2*a_N))*term_3)
    log_t_unnormalized = log_t_unnormalized + -0.5 * torch.log((b_N/a_N)*torch.det(torch.inverse(Λ_N)))

    term_3 = -0.5 * d * torch.log(2*a_N)
    term_4 = -0.5 * d * torch.log(torch.tensor(torch.pi))
    log_t_unnormalized = log_t_unnormalized + term_3 + term_4

    return log_t_unnormalized


def train_with_student_t(flows, d, X, Y, epoch, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows with Student-t likelihood")
    losses = []
    for i in range(epoch):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_likelihood = vectorized_log_t_distribution_unnormalized(q_samples, d, X, Y)
        log_p = log_likelihood
        loss = torch.mean(q_log_prob - log_p)
        if i == 0 or i % 100 == 0 or i + 1 == epoch:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def generate_synthetic_data(d, n):
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    # W = torch.tensor([2.5, 0.4])
    v = torch.tensor(0.5)
    noise = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + noise

    n_test = 3
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


def compute_posterior_t_distribution_parameters(X, y):
    N = len(X)
    d = X[0].shape[0]
    dimension = X[0].shape[0]
    μ_0 = torch.zeros(dimension)
    cov_0 = torch.eye(dimension)
    a_0 = torch.tensor(N)
    b_0 = torch.tensor(N)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), y)))
    a_N = a_0 + (d / 2.)
    b_N = b_0 + 0.5 * (torch.matmul(y.t(), y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_N.t(), Λ_N), μ_N))

    mean = μ_N
    scale_matrix = (b_N / a_N) * torch.inverse(Λ_N)
    df = 2 * a_N
    return mean, scale_matrix, df


def compute_posterior_predictive_t_distribution_parameters(X_train, y_train, X_pred):
    N = len(X_train)
    d = X_train[0].shape[0]
    μ_0 = torch.zeros(d)
    cov_0 = torch.eye(d)
    a_0 = torch.tensor(N)
    b_0 = torch.tensor(N)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X_train.t(), X_train) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X_train.t(), y_train)))
    a_N = a_0 + (d / 2.)
    b_N = b_0 + 0.5 * (torch.matmul(y_train.t(), y_train) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_0.t(), Λ_N), μ_N))

    mean = torch.matmul(X_pred, μ_N)
    scale_matrix = (b_N / a_N) * (torch.eye(len(X_pred)) + torch.matmul(torch.matmul(X_pred, Λ_N), X_pred.t()))
    df = 2 * a_N
    return mean, scale_matrix, df


def posterior( X_train, Y_train, W, dimension):
    num_iter = 2000
    q_sample_size = 100

    flows = build_flow_model(dimension)
    flows, losses = train_with_student_t(flows, dimension, X_train, Y_train, num_iter, q_sample_size)
    # View.plot_loss(losses)

    q_samples = flows.sample(10000)
    fixed = W.tolist()
    sample_mean = torch.mean(q_samples, dim=0).tolist()
    for i in range(dimension):
        print(f"Index {i}: {fixed[i]} Student-t W: {sample_mean[i]}")

    mean, scale_matrix, df = compute_posterior_t_distribution_parameters(X_train, Y_train)
    View.plot_analytical_flow_posterior_t_distribution_on_grid(dimension, mean, scale_matrix, df, flows)


def posterior_predictive(X_train, Y_train, X_test, Y_test):
    num_iter = 2000
    y_sample_size = 100
    y_length = Y_test.shape[0]
    flows = build_flow_model(y_length)

    # To complete --> Train posterior predictive

    mean, scale_matrix, df = compute_posterior_predictive_t_distribution_parameters(X_train, Y_train, X_test)
    View.plot_analytical_flow_posterior_predictive_t_distribution_on_grid(len(X_test), mean, scale_matrix, df, flows)
    print(mean)
    print(Y_test)
    return


def main():
    # Generate some real world samples X
    dimension = 3
    num_samples = 100
    X_train, Y_train, W, variance, X_test, Y_test = generate_synthetic_data(dimension, num_samples)

    posterior(X_train, Y_train, W, dimension)



if __name__ == "__main__":
    main()

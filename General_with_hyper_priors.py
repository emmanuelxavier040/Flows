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
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift
from enflows.transforms.lu import LULinear
from enflows.transforms.svd import SVDLinear

import Visualizations as View


def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=1)
    return log_likelihood


def vectorized_log_likelihood_t_distribution_unnormalized(Ws, d, X, Y):
    a_0 = len(X) / 2
    b_0 = a_0
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    term_1 = -(a_0 + d / 2)
    term_2 = torch.log(1 + (1 / (2 * b_0)) * torch.sum(squared_errors, dim=1))
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
    log_prior = vectorized_log_prior_unnormalized(q_samples, variance)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train(flows, d, X, Y, variance, num_iter, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    for i in range(num_iter):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, variance)
        loss = torch.mean(q_log_prob - log_p)
        if (i + 1) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        loss.backward()
        optimizer.step()
    return flows


def train_with_student_t(flows, d, X, Y, variance, num_iter, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    for i in range(num_iter):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_likelihood = vectorized_log_likelihood_t_distribution_unnormalized(q_samples, d, X, Y)
        log_prior = vectorized_log_prior_unnormalized(q_samples, variance)
        log_p = log_likelihood + log_prior
        # print(log_likelihood, "============")
        # print(log_prior)
        loss = torch.mean(q_log_prob - log_p)
        if (i + 1) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        loss.backward()
        optimizer.step()
    return flows


def sample_Ws(model, sample_size, min_lambda, max_lambda, variance, interval_size, n_iter):
    sample_list, lambda_list, loss_list = [], [], []
    with torch.no_grad():
        for _ in range(n_iter):
            uniform_lambdas = torch.rand(interval_size).cuda()
            lambdas_exp = (uniform_lambdas * (max_lambda - min_lambda) + min_lambda).view(-1, 1)
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)
            posterior_eval = vectorized_log_posterior_unnormalized(q_samples, X, Y, variance)


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
    return X, Y, W, v


def build_flow_model(d):
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 15
    for _ in range(num_layers):
        # InverseTransform(transforms.append(LULinear(features=d)))
        # InverseTransform(transforms.append(SVDLinear(features=d, num_householder=4)))
        InverseTransform(transforms.append(NaiveLinear(features=d)))
        InverseTransform(transforms.append(ScalarScale(scale=1.5)))
        InverseTransform(transforms.append(ScalarShift(shift=0.5)))
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


# Define a Multivariate-Normal distribution and generate some real world samples X
dimension = 2
num_samples = 100
X, Y, W, variance = generate_synthetic_data(dimension, num_samples)
flows = build_flow_model(dimension)

num_iter = 500
q_sample_size = 100
flows = train(flows, dimension, X, Y, variance, num_iter, q_sample_size)
# q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
# log_p = vectorized_log_posterior_unnormalized(q_samples, X, Y, variance)
q_samples = flows.sample(1000)
sample_mean_1 = torch.mean(q_samples, dim=0).tolist()
fixed = W.tolist()

# flows = train_with_student_t(flows, dimension, X, Y, variance, num_iter, q_sample_size)
# q_samples = flows.sample(1000)
sample_mean = torch.mean(q_samples, dim=0).tolist()
for i in range(dimension):
    print(f"Index {i}: {fixed[i]} Fixed Sigma W: {sample_mean_1[i]} Student-t W: {sample_mean[i]}")

analytical_mean, analytical_cov = compute_analytical_posterior_for_fixed_variance(X, Y, torch.zeros(dimension),
                                                                                  torch.eye(dimension), variance)
view_samples = q_samples.detach().numpy()
# View.plot_analytical_vs_flow_posterior_ridge_regression_fixed_variance(analytical_mean, analytical_cov, view_samples)
View.plot_analytical_vs_flow_posterior_ridge_regression_fixed_variance_2(analytical_mean, analytical_cov, flows)

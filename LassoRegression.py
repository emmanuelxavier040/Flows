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
import numpy as np

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift

import Visualizations as View


def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=1)
    return log_likelihood


def vectorized_log_likelihood_t_distribution_unnormalized(Ws, d, X, Y):
    a_0 = len(X)/2
    b_0 = a_0
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    term_1 = -(a_0 + d/2)
    term_2 = torch.log(1 + (1/(2 * b_0)) * torch.sum(squared_errors, dim=1))
    log_likelihood = term_1 * term_2
    return log_likelihood


def vectorized_log_prior_unnormalized(Ws, d, lambdas_list, variance):
    std_dev = torch.sqrt(variance)
    term_1 = d * torch.log(lambdas_list/std_dev)
    term_2 = -(lambdas_list/std_dev)*torch.norm(Ws, p=1, dim=1)
    log_prior = term_1 + term_2
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_list, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    log_prior = vectorized_log_prior_unnormalized(q_samples, d, lambdas_list, variance)
    log_posterior = log_likelihood + log_prior
    # print(log_likelihood, "------------------")
    # print(log_prior)
    return log_posterior


def train(flows, d, X, Y, variance, num_iter, q_sample_size, lamda = 1.0):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        lambdas_list = torch.ones(q_sample_size) * lamda
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_list, variance)
        loss = torch.mean(q_log_prob - log_p)
        if (i + 1) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def train_2(flows, d, X, Y, variance, num_iter, q_sample_size, lamda = 1.0):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        lambdas_list = torch.ones(q_sample_size) * lamda
        log_likelihood = vectorized_log_likelihood_t_distribution_unnormalized(q_samples, d, X, Y)
        log_prior = vectorized_log_prior_unnormalized(q_samples, d, lambdas_list, variance)
        log_p = log_likelihood + log_prior
        loss = torch.mean(q_log_prob - log_p)
        if (i + 1) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def sample_Ws(model, sample_size, d, min_lambda, max_lambda, variance, interval_size, n_iter = 5):
    view_q_samples_list, view_lambda_list, view_loss_list = [], [], []
    with torch.no_grad():
        for _ in range(n_iter):
            uniform_lambdas = torch.rand(q_sample_size)
            lambda_list = (uniform_lambdas * (max_lambda - min_lambda) + min_lambda).view(-1, 1)
            uniform_p = lambda_list.new_ones(q_sample_size).view(-1, 1)
            context = torch.cat((lambda_list, uniform_p), 1)
            q_samples_list, log_probs_samples = flows.sample_and_log_prob(q_sample_size, context)
            for q_samples, lamda in zip(q_samples_list, lambda_list):
                log_posteriors = vectorized_log_prior_unnormalized(q_samples, d, lamda, variance)
                loss = log_probs_samples - log_posteriors
                view_q_samples_list.append(q_samples.detach())
                view_lambda_list.append(lamda.detach())
                view_loss_list.append(loss.detach())

    view_lambda_list = np.concatenate(view_lambda_list, 0)
    view_lambda_sorted_idx = view_lambda_list.argsort()
    view_lambda_list_sorted = view_lambda_list[view_lambda_sorted_idx]
    view_loss_list_sorted = np.array(view_loss_list)[view_lambda_sorted_idx]
    view_q_samples_list_sorted = np.array(view_q_samples_list)[view_lambda_sorted_idx]
    return view_q_samples_list_sorted, view_lambda_list_sorted, view_loss_list_sorted


def generate_synthetic_data(d, l, n, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    W = torch.tensor([2.3, 4.5])
    # W [-l:] = 0
    v = torch.tensor(noise)
    delta = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


def build_flow_model(d):
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
    for _ in range(num_layers):
        InverseTransform(transforms.append(NaiveLinear(features=d)))
        InverseTransform(transforms.append(ScalarScale(scale=2)))
        InverseTransform(transforms.append(ScalarShift(shift=1.5)))
    transforms = transforms[::-1]
    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


lamda = 1
dimension, last_zero_indices = 2, 2
data_sample_size = 30
noise = 0.1
print(f"============= Parameters ======== \n"
      f"lambda:{lamda}, dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
      f"num_samples:{data_sample_size}, noise:{noise}\n")

X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, data_sample_size, noise)
flows = build_flow_model(dimension)

num_iter = 1000
q_sample_size = 100
flows, losses = train(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)
View.plot_loss(losses)


q_samples = flows.sample(100)
sample_mean_1 = torch.mean(q_samples, dim=0).tolist()
fixed = W.tolist()

flows = build_flow_model(dimension)
flows, losses = train_2(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)
q_samples = flows.sample(100)
view_samples = q_samples.detach().numpy()
View.plot_mvn_2(view_samples)
sample_mean = torch.mean(q_samples, dim=0).tolist()

for i in range(dimension):
    print(f"Index {i}: {fixed[i]} Fixed sigma W: {sample_mean_1[i]} Student-t W: {sample_mean[i]}")


# w_samples_list, lambda_list, loss_list = sample_Ws(flows, q_sample_size, dimension, 1, 10, variance, 10)
# print(w_samples_list)
# print(lambda_list)
# print(loss_list)
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
from enflows.nn.nets import ResidualNet, Sin
from enflows.transforms import iResBlock
from enflows.transforms.conditional import ConditionalShiftTransform, ConditionalScaleTransform, ConditionalLUTransform

import Visualizations as View


def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=-1)
    # print(squared_errors)
    # print(torch.sum(squared_errors, dim=2))
    # print("Log likelihood: ==>", log_likelihood)
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


def vectorized_log_prior_unnormalized(Ws, d, lambdas_exp, variance):
    lambdas_list = 10**lambdas_exp
    std_dev = torch.sqrt(variance)
    term_1 = d * torch.log(lambdas_list / std_dev)
    # print(lambdas_list)
    # print(torch.norm(Ws, p=1, dim=2) / std_dev)
    # print((lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=2))
    term_2 = -(lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    # print("Log Prior: ==> ", log_prior)
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    log_prior = vectorized_log_prior_unnormalized(q_samples, d, lambdas_exp, variance)
    # print(log_likelihood, "------------------")
    # print(log_prior)
    log_posterior = log_likelihood + log_prior
    # log_posterior = log_prior
    # print(log_posterior)
    return log_posterior


def train(flows, d, X, Y, variance, num_iter, q_sample_size, lamda=1.0):
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


def train_conditional_flows(flows, d, X, Y, variance, num_iter, q_sample_size):
    context_size = 1
    lambda_min_exp = -3
    lambda_max_exp = 3

    optimizer = optim.Adam(flows.parameters(), lr=1e-4)
    print("Starting training the flows")
    losses = []

    for i in range(num_iter):
        optimizer.zero_grad()
        uniform_lambdas = torch.rand(context_size)
        lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
        context = lambdas_exp
        # print("Context : ", context)
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size, context=context)
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, variance)
        loss = torch.mean(q_log_prob - log_p)
        # print(loss.tolist())
        if (i) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def train_2(flows, d, X, Y, variance, num_iter, q_sample_size, lamda=1.0):
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


def generate_synthetic_data(d, l, n, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    # W[-l:] = 0
    v = torch.tensor(noise)
    delta = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


def build_flow_model(d):
    context_features = 16
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
    for _ in range(num_layers):
        InverseTransform(transforms.append(NaiveLinear(features=d)))
        InverseTransform(transforms.append(ScalarScale(scale=2)))
        InverseTransform(transforms.append(ScalarShift(shift=1.5)))
    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


def build_conditional_flow_model(d):
    context_features = 4
    print("Defining the flows")

    densenet_factory = iResBlock.Factory()
    densenet_factory.set_logabsdet_estimator(brute_force=True,  # set this to false for high dimensions (>3)
                                             # unbiased_estimator=True,  # default;
                                             # trace_estimator="neumann"  # either "neumann" or "basic";
                                             )
    densenet_factory.set_densenet(condition_input=True,
                                  condition_lastlayer=False,
                                  condition_multiplicative=True,
                                  ###
                                  dimension=d,
                                  densenet_depth=2,
                                  densenet_growth=16,
                                  c_embed_hidden_sizes=(64, 64, 10),
                                  m_embed_hidden_sizes=(64, 64),
                                  activation_function=Sin(10),
                                  lip_coeff=.97,
                                  context_features=context_features)

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
    for _ in range(num_layers):
        InverseTransform(transforms.append(ConditionalLUTransform(features=d, hidden_features=64,
                                                                  context_features=context_features)))
        InverseTransform(transforms.append(ConditionalScaleTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
        InverseTransform(transforms.append(ConditionalShiftTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
        # InverseTransform(transforms.append(densenet_factory.build()))
    # transforms = transforms[::-1]
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=32,
                                num_blocks=5, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


lamda = 1
dimension, last_zero_indices = 3, 3
data_sample_size = 100
noise = 0.1
print(f"============= Parameters ======== \n"
      f"lambda:{lamda}, dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
      f"num_samples:{data_sample_size}, noise:{noise}\n")

X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, data_sample_size, noise)
num_iter = 500
q_sample_size = 1

# train flows for fixed lambda
# flows, losses = train(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)

# train conditional flows
# flows = build_flow_model(dimension)
flows = build_conditional_flow_model(dimension)
flows, losses = train_conditional_flows(flows, dimension, X, Y, variance, num_iter, q_sample_size)

# flows, losses = train(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)
View.plot_loss(losses)
#
#
# uniform_lambdas = torch.rand(10)
# lambdas_exp = (uniform_lambdas * (1 - -2) + -2).view(-1, 1)
# context = lambdas_exp
# q_samples, log_ps = flows.sample_and_log_prob(100, context)
# print(q_samples)
# print(log_ps)

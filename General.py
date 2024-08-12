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

import Visualizations as View


def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=1)
    return log_likelihood


def vectorized_log_prior_unnormalized(Ws, variance):
    outputs = torch.matmul(Ws.unsqueeze(1), Ws.unsqueeze(-1))
    squared_weights = outputs.squeeze()
    log_prior = -0.5 * (1 / variance) * squared_weights
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, X, Y, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    log_prior = vectorized_log_prior_unnormalized(q_samples, variance)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train(flows, X, Y, variance, num_iter, q_sample_size):
    optimizer = optim.Adam(flows.parameters())
    print("Starting training the flows")
    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
        log_p = vectorized_log_posterior_unnormalized(q_samples, X, Y, variance)
        loss = torch.mean(q_log_prob - log_p)
        if (i + 1) % 100 == 0:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    return flows, losses


def generate_synthetic_data(d, l, n):
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    W[-l:] = 0
    v = torch.tensor(1)
    noise = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + noise
    # print(X, Y, W, v)
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
    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


def main():
    # Define a Multivariate-Normal distribution and generate some real world samples X
    dimension = 40
    last_zero_indices = 5
    num_samples = 50
    X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, num_samples)

    flows = build_flow_model(dimension)

    num_iter = 1000
    q_sample_size = 100
    flows, losses = train(flows, X, Y, variance, num_iter, q_sample_size)
    View.plot_loss(losses)

    q_samples = flows.sample(100)
    sample_mean = torch.mean(q_samples, dim=0).tolist()
    fixed = W.tolist()

    # print("Parameter W used : ", fixed[:20])
    # print("Mean value of W learned by flows : ", sample_mean[:20])
    for i in range(dimension):
        print(f"Index {i}: {fixed[i]} - {sample_mean[i]}")


if __name__ == "__main__":
    main()

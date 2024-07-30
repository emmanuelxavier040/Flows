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


def log_likelihood_unnormalized(W, X, Y, variance):
    # proportional to p(Samples|q) where p(Samples|q) = N(XW, variance)
    log_likelihood = -0.5 * (1/variance) * torch.sum((Y - torch.matmul(X, W)) ** 2)
    return log_likelihood


def log_prior_unnormalized(W, variance):
    # proportional to p(q) where p(q) = N(0, variance)
    log_prior = -0.5 * (1/variance) * torch.sum(W ** 2)
    return log_prior


def log_posterior_unnormalized(q, samples_X, samples_Y, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = log_likelihood_unnormalized(q, samples_X, samples_Y, variance)
    log_prior = log_prior_unnormalized(q, variance)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def calculate_loss(q_samples, q_log_prob, samples_X, samples_Y, variance):
    losses = []
    for q, log_prob in zip(q_samples, q_log_prob):
        log_posterior = log_posterior_unnormalized(q, samples_X, samples_Y, variance)
        losses.append(log_prob - log_posterior)

    kl_divergence = torch.mean(torch.stack(losses))
    return kl_divergence


# Define a Multivariate-Normal distribution and generate some real world samples X
print("Generating real-world samples")
d = 50
data_mean = torch.randn(1, d)
data_cov = torch.eye(d)
mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
num_data_samples = torch.Size([100])
X = mvn_dist.sample(num_data_samples)

# Define a parameter which can be used for a linear transformation of X to Y
W = torch.randn(d)

# Define noise that is added to XW to make Y noisy.
variance = torch.tensor(0.5, dtype=torch.float32)
noise = torch.randn(num_data_samples) * variance
Y = torch.matmul(X, W) + noise
# print(X, Y, W)
print("Defining the flows")
base_dist = StandardNormal(shape=[d])
transforms = []
num_layers = 5
for _ in range(num_layers):
    InverseTransform(transforms.append(NaiveLinear(features=d)))
    InverseTransform(transforms.append(ScalarScale(scale=2)))
    InverseTransform(transforms.append(ScalarShift(shift=1.5)))
transform = CompositeTransform(transforms)
flow = Flow(transform, base_dist)

optimizer = optim.Adam(flow.parameters())
num_iter = 1000
q_sample_size = 100
print("Starting training the flows")
for i in range(num_iter):
    optimizer.zero_grad()
    q_samples, q_log_prob = flow.sample_and_log_prob(q_sample_size)
    loss = calculate_loss(q_samples, q_log_prob, X, Y, variance)
    if (i+1) % 100 == 0:
        print("Loss after iteration {}: ".format(i), loss.tolist())
    loss.backward()
    optimizer.step()

q_samples = flow.sample(100)
print("Parameter W used : ", W.tolist())
print("Mean value of W learned by flows : ", torch.mean(q_samples, dim=0).tolist())

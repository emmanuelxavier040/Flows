
import torch
from torch import optim
import numpy as np

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift
from enflows.transforms import ActNorm

import Visualizations as View

torch.manual_seed(11)
np.random.seed(10)

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


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


def sum_of_norms_of_W_groups(Ws, indices_list):
    result_tensors = []
    group_norms = []
    sum_tensor = torch.zeros(len(Ws))
    for indices in indices_list:
        result_tensors.append(Ws[:, indices])
        group_norms.append(torch.norm(Ws[:, indices], p=2, dim=1))
        sum_tensor = sum_tensor + torch.norm(Ws[:, indices], p=2, dim=1)
    return sum_tensor


def vectorized_log_prior_unnormalized(Ws, d, lambdas_list, variance):
    sum_tensor = sum_of_norms_of_W_groups(Ws, indices_list)
    std_dev = torch.sqrt(variance)
    log_prior = -(lambdas_list/std_dev)*sum_tensor
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_list, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    log_prior = vectorized_log_prior_unnormalized(q_samples, d, lambdas_list, variance)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train(flows, d, X, Y, variance, num_iter, q_sample_size, lamda = 1):
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


def train_2(flows, d, X, Y, variance, num_iter, q_sample_size, lamda = 1):
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


def generate_synthetic_data(d, n, l, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    W = torch.rand(d)
    W[-l:] = 0
    # W = torch.tensor([50.2, 30.7, 29.4, 4.3, 4.3])
    v = torch.tensor(noise)
    delta = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


def build_flow_model(d):
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 10
    for _ in range(num_layers):
        transforms.append(InverseTransform(NaiveLinear(features=d)))
        transforms.append(InverseTransform(ScalarScale(scale=2)))
        transforms.append(InverseTransform(ScalarShift(shift=1.5)))
        transforms.append(InverseTransform(ActNorm(features=d)))

    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


def main():
    lamda = 0.5
    dimension = 10
    # indices_list = [[0, 2], [1, 3, 4], [5, 6, 7, 8, 9], [i for i in range(10, 49)]]
    indices_list = [[0, 2], [1, 3, 4], [5, 6, 7, 8, 9]]
    data_sample_size = 50
    last_zero_indices = 5
    noise = 0.05
    print(f"============= Parameters ======== \n"
          f"lambda:{lamda}, dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
          f"num_samples:{data_sample_size}, noise:{noise}\n")

    X, Y, W, variance = generate_synthetic_data(dimension, data_sample_size, last_zero_indices, noise)

    flows = build_flow_model(dimension)

    num_iter = 1000
    q_sample_size = 100
    flows, losses = train(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)
    # View.plot_loss(losses)
    # q_samples, q_log_prob = flows.sample_and_log_prob(q_sample_size)
    # lambdas_list = torch.ones(q_sample_size)
    # log_p = vectorized_log_posterior_unnormalized(q_samples, dimension, X, Y, lambdas_list, variance)
    q_samples = flows.sample(100)
    sample_mean_1 = torch.mean(q_samples, dim=0).tolist()
    fixed = W.tolist()

    flows, losses = train_2(flows, dimension, X, Y, variance, num_iter, q_sample_size, lamda)
    q_samples = flows.sample(100)
    sample_mean = torch.mean(q_samples, dim=0).tolist()

    for i in range(dimension):
        print(f"Index {i}: {fixed[i]} - {sample_mean_1[i]} - {sample_mean[i]}")


if __name__ == "__main__":
    main()

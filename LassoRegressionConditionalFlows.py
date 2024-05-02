"""
This is an example of posterior estimation using flows by minimizing the KL divergence with synthetic data.
We use a Multi-variate normal distribution to generate X. We choose a fixed parameter W which can be used for a linear
transformation of X to Y. We can add some noise to the observations finally giving Y = XW + noise. Our task is to infer
about the posterior P(W | X,Y). We use linear flows to compute the KL-Divergence(q(W) || P*(W | X,Y)). Here P* is the
un-normalized posterior which is equivalent to the un-normalized Gaussian likelihood * Gaussian prior.
We can compute P* since I have X, Y and W (for W, we can easily sample from flow). After training, flows should have
learned the distribution of W and samples from it should resemble the fixed W which we used to transform X to Y.
"""
import os.path

import torch
from torch import optim
import numpy as np
import keyboard


from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift
from enflows.nn.nets import ResidualNet
from enflows.transforms.conditional import ConditionalShiftTransform, ConditionalScaleTransform, ConditionalLUTransform

import LassoRegression
import Visualizations as View

torch.manual_seed(20)

def vectorized_log_likelihood_unnormalized(Ws, X, Y, variance):
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    log_likelihood = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=-1)
    return log_likelihood


def vectorized_log_likelihood_t_distribution_unnormalized(Ws, d, X, Y):
    a_0 = len(X) / 2
    b_0 = a_0
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    term_1 = -(a_0 + d / 2)
    term_2 = torch.log(1 + (1 / (2 * b_0)) * torch.sum(squared_errors, dim=-1))
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


def train(flows, d, X, Y, variance, epochs, n, fixed_lambda_exp=torch.tensor([1.0])):
    optimizer = optim.Adam(flows.parameters(), lr=1e-3)
    print("Starting training the flows")
    losses = []
    lambdas_exp = fixed_lambda_exp.view(-1, 1)
    context = lambdas_exp
    for i in range(epochs):
        optimizer.zero_grad()
        q_samples, q_log_prob = flows.sample_and_log_prob(n, context=context)
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, variance)
        loss = torch.mean(q_log_prob - log_p)
        if i == 0 or i % 100 == 0 or i + 1 == epochs:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()

    return flows, losses


def train_conditional_flows(flows, d, X, Y, variance, epochs, n, context_size=100,
                            lambda_min_exp=-1, lambda_max_exp=2):
    file_name = f'CNF_d{d}_n{n}_e{epochs}_lmin{lambda_min_exp}_lmax{lambda_max_exp}'

    optimizer = optim.Adam(flows.parameters(), lr=1e-3)
    print("Starting training the flows")
    losses = []
    torch.set_anomaly_enabled(True)
    for i in range(epochs):
        optimizer.zero_grad()
        uniform_lambdas = torch.rand(context_size)
        lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
        context = lambdas_exp
        q_samples, q_log_prob = flows.sample_and_log_prob(n, context=context)
        log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, variance)
        loss = torch.mean(q_log_prob - log_p)
        # print(loss.tolist())
        if i % 10 == 0 or i + 1 == epochs:
            print("Loss after iteration {}: ".format(i), loss.tolist())
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        if keyboard.is_pressed('esc'):
            print("Stopping training... ")
            break

    save_model(flows, file_name)

    return flows, losses


def load_model(dimensions, model_path):
    model = build_conditional_flow_model(dimensions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(model, file_name):
    folder_name = "./models/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(model.state_dict(), f"{folder_name}Flows_{file_name}")


def generate_synthetic_data(d, l, n, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.randn(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    # W = torch.rand(d) * 20 - 10
    W = torch.randn(d)
    print(W)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    if l < d:
        W[-l:] = 0
    v = torch.tensor(noise)
    delta = torch.randn(num_data_samples) * v
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


def build_flow_model(d):
    print("Defining the flows")
    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 15
    for _ in range(num_layers):
        InverseTransform(transforms.append(NaiveLinear(features=d)))
        InverseTransform(transforms.append(ScalarScale(scale=2)))
        InverseTransform(transforms.append(ScalarShift(shift=1.5)))
    transform = CompositeTransform(transforms)
    model = Flow(transform, base_dist)
    return model


def build_conditional_flow_model(d):
    context_features = 16
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 10
    for _ in range(num_layers):
        InverseTransform(transforms.append(ConditionalLUTransform(features=d, hidden_features=64,
                                                                  context_features=context_features)))
        InverseTransform(transforms.append(ConditionalScaleTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
        InverseTransform(transforms.append(ConditionalShiftTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=32,
                                num_blocks=5, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def view_original_vs_flow_learnt_parameters(d, fixed, flows, context=None):
    sample_size = 10000
    sample_mean = None
    if context is None:
        q_samples = flows.sample(sample_size)
        sample_mean = torch.mean(q_samples, dim=0).tolist()
    else:
        q_samples = flows.sample(sample_size, context=context.view(-1, 1))
        sample_mean = torch.mean(q_samples[0], dim=0).tolist()

    print(f"Index ||  Original  ||  Fixed Lambda ")
    for i in range(d):
        print(f"Index {i}       :     {fixed[i]}      :       {sample_mean[i]}")


def main():
    dimension, last_zero_indices = 8, 8
    data_sample_size = 20
    noise = 0.1
    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
          f"Sample Size:{data_sample_size}, noise:{noise}\n")

    X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, data_sample_size, noise)
    original_W = W.tolist()
    variance = torch.tensor(0.3)
    epochs = 1000
    q_sample_size = 1

    fixed_lambda_exp = torch.rand(1)
    print("Fixed Lambda exponent: ", fixed_lambda_exp)
    #==================================================================
    # train flows for fixed lambda with unconditional version
    flows = build_flow_model(dimension)
    flows, losses = LassoRegression.train(flows, dimension, X, Y, variance, epochs,
                                          q_sample_size, fixed_lambda_exp.item())
    view_original_vs_flow_learnt_parameters(dimension, original_W, flows)


    #==================================================================
    # train flows for fixed lambda with CNF
    flows = build_flow_model(dimension)
    flows, losses = train(flows, dimension, X, Y, variance, epochs, q_sample_size, fixed_lambda_exp)
    view_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)


    #==================================================================
    # train conditional flows
    context_size = 5000
    lambda_min_exp = -1
    lambda_max_exp = 5
    flows = build_conditional_flow_model(dimension)
    flows, losses = train_conditional_flows(flows, dimension, X, Y, variance, epochs, q_sample_size,
                                            context_size, lambda_min_exp, lambda_max_exp)
    view_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)

    # View.plot_loss(losses)
    View.plot_lasso_beta_vs_lambda(dimension, flows, lambda_min_exp, lambda_max_exp)


if __name__ == "__main__":
    main()
# device = "cuda:0" if torch.cuda.is_available() else 'cpu'

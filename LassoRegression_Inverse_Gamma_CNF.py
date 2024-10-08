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

import numpy as np
import scipy as sp
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import ConditionalShiftTransform, ConditionalScaleTransform, ConditionalLUTransform, \
    ConditionalSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm

from torch import optim

import Visualizations as View

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def vectorized_log_likelihood_unnormalized(Ws, X, Y, likelihood_sigma):
    variance = torch.tensor(likelihood_sigma) ** 2
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    n = Y.shape[0]
    term_1 = -0.5 * n * torch.log(2 * torch.pi * variance)
    term_2 = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=-1)
    log_likelihood = term_1 + term_2
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


def vectorized_log_prior_unnormalized(Ws, d, lambdas_exp, sigma):
    variance = torch.tensor(sigma) ** 2
    lambdas_list = 10 ** lambdas_exp
    std_dev = torch.sqrt(variance)
    term_1 = d * torch.log(lambdas_list / (2 * std_dev))
    term_2 = -(lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    return log_prior


def vectorized_standard_laplace_log_prior_unnormalized(Ws, d, lambdas_exp, n):
    lambdas_list = (10 ** lambdas_exp)
    term_1 = d * torch.log(lambdas_list / 2.0)
    term_2 = -lambdas_list * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, likelihood_sigma):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, likelihood_sigma)
    # log_likelihood = vectorized_log_likelihood_t_distribution_unnormalized(q_samples, d, X, Y)
    # log_prior = vectorized_log_prior_unnormalized(q_samples, d, lambdas_exp, likelihood_sigma)
    log_prior = vectorized_standard_laplace_log_prior_unnormalized(q_samples, d, lambdas_exp, Y.shape[0])
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train_CNF(flows, d, X, Y, likelihood_sigma, epochs, n, context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
    file_name = f'CNF_d{d}_n{n}_e{epochs}_lmin{lambda_min_exp}_lmax{lambda_max_exp}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)

    print("Starting training the flows")
    losses = []

    try:
        for epoch in range(epochs):
            # print("==================================")
            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = lambdas_exp
            q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=n, context=context)
            log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, likelihood_sigma)
            loss = torch.mean(q_log_prob - log_p)
            loss.backward()

            if epoch % 10 == 0 or epoch + 1 == epochs:
                print("Loss after iteration {}: ".format(epoch), loss.tolist())
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(flows.parameters(), 1)
            optimizer.step()


    except KeyboardInterrupt:
        print("interrupted..")

    # save_model(flows, file_name)

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
    # Define a Multivariate-Normal distribution and generate some real world samples X and Y
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))
    data_mean = torch.zeros(d)
    data_cov = torch.eye(d)
    data_mvn_dist = torch.distributions.MultivariateNormal(data_mean, data_cov)
    num_data_samples = torch.Size([n])
    X = data_mvn_dist.sample(num_data_samples)
    # W = torch.rand(d) * 20 - 10
    W = torch.randn(d)

    min_val = torch.min(W)
    max_val = torch.max(W)
    W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    if l < d:
        W[-l:] = 0
    v = torch.tensor(noise ** 2)
    delta = torch.randn(num_data_samples) * v
    # delta = torch.normal(0, noise ** 2, num_data_samples)
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


def generate_regression_dataset(n_samples, n_features, n_non_zero, noise_std):
    assert n_features >= n_non_zero

    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    coefficients[non_zero_indices] = np.random.normal(0, 1, n_non_zero)

    scale_matrix = np.eye(n_features)
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)

    y = np.dot(X, coefficients) + np.random.normal(0, noise_std ** 2,
                                                   n_samples)

    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(coefficients).float()



def build_conditional_flow_model(d):
    context_features = 16
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 10
    for _ in range(num_layers):
        transforms.append(InverseTransform(ConditionalLUTransform(features=d, hidden_features=64,
                                                                  context_features=context_features)))
        transforms.append(InverseTransform(ConditionalScaleTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
        transforms.append(InverseTransform(ConditionalShiftTransform(features=d, hidden_features=64,
                                                                     context_features=context_features)))
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=64,
                                num_blocks=1, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


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


def build_masked_sigmoid_conditional_flow_model(d):
    context_features = 8
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 3
    for _ in range(num_layers):
        transforms.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(
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
                                num_blocks=5, activation=torch.nn.functional.relu)
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


def sample_Ws_for_plots(flows, X, Y, likelihood_sigma, context_size, flow_sample_size, lambda_min_exp, lambda_max_exp):
    d = X.shape[1]
    num_iter = 10
    lambdas, q_samples_list, losses = [], [], []

    with torch.no_grad():
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            q_samples, q_log_probs = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_p_samples = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, likelihood_sigma)
            loss = q_log_probs - log_p_samples

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


def main():
    # Set the parameters
    epochs = 10000
    dimension, last_zero_indices = 5, 20
    data_sample_size = 7
    data_noise_sigma = 2.0
    likelihood_sigma = 2
    q_sample_size = 1
    context_size = 1000
    lambda_min_exp = -3
    lambda_max_exp = 2

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    # X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, data_sample_size, data_noise_sigma)

    X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    X /= X.std(0)

    X_torch = X.to(device)
    Y_torch = Y.to(device)

    original_W = W.tolist()
    print("Original Parameters: ", original_W)
    fixed_lambda_exp = torch.rand(1)
    print("Fixed Lambda exponent: ", fixed_lambda_exp)

    # ==================================================================
    # train conditional flows

    # flows = build_conditional_flow_model(dimension)
    # flows = build_masked_sigmoid_conditional_flow_model(dimension)
    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows.to(device)

    learning_rate = 1e-3
    flows, losses = train_CNF(flows, dimension, X_torch, Y_torch,
                              likelihood_sigma, epochs,
                              q_sample_size,
                              context_size, lambda_min_exp, lambda_max_exp,
                              learning_rate
                              )
    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    # View.plot_loss(losses)
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Y_torch,
                                                                          likelihood_sigma, 100, 100,
                                                                          lambda_min_exp, lambda_max_exp)
    View.plot_flow_lasso_path_vs_ground_truth(X, Y,
                                              lambdas_sorted, q_samples_sorted, likelihood_sigma ** 2)

    title = "Lasso-Regression-CNF"
    View.plot_log_marginal_likelihood_vs_lambda(X, Y, lambdas_sorted, losses_sorted, likelihood_sigma ** 2, title)
    View.plot_lasso_path_variance(lambdas_sorted, q_samples_sorted)


if __name__ == "__main__":
    main()

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

import scipy as sp
import torch
from sklearn.linear_model import LinearRegression
from torch import optim
import numpy as np

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift
from enflows.nn.nets import ResidualNet
from enflows.transforms.conditional import ConditionalShiftTransform, ConditionalScaleTransform, ConditionalLUTransform, \
    ConditionalSumOfSigmoidsTransform


import LassoGroundTruth
import LassoRegression
import Visualizations as View
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm

torch.manual_seed(11)
np.random.seed(10)


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
    lambdas_list = 10 ** lambdas_exp
    std_dev = torch.sqrt(variance)
    term_1 = d * torch.log(lambdas_list / (2 * std_dev))
    # print(lambdas_list)
    # print(torch.norm(Ws, p=1, dim=2) / std_dev)
    # print((lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=2))
    term_2 = -(lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    # print("Log Prior: ==> ", log_prior)
    return log_prior


def vectorized_standard_laplace_log_prior_unnormalized(Ws, d, lambdas_exp, variance):
    lambdas_list = 10 ** lambdas_exp
    term_1 = d * torch.log(lambdas_list * 0.5)
    # print(lambdas_list)
    # print(torch.norm(Ws, p=1, dim=2) / std_dev)
    # print((lambdas_list / std_dev) * torch.norm(Ws, p=1, dim=2))
    term_2 = -lambdas_list * torch.norm(Ws, p=1, dim=-1)
    log_prior = term_1 + term_2
    # print("Log Prior: ==> ", log_prior)
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, variance):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, variance)
    log_prior = vectorized_standard_laplace_log_prior_unnormalized(q_samples, d, lambdas_exp, variance)
    log_posterior = log_likelihood + log_prior
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
        if i == 0 or i % 10 == 0 or i + 1 == epochs:
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
    try:
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


def sample_Ws(model, sample_size, d, context_size, min_lambda, max_lambda, variance, n_iter=1):
    view_q_samples_list, view_lambda_list, view_loss_list, view_log_probs = [], [], [], []
    with torch.no_grad():
        for _ in range(n_iter):
            uniform_lambdas = torch.rand(context_size)
            lambda_list = (uniform_lambdas * (max_lambda - min_lambda) + min_lambda).view(-1, 1)
            context = lambda_list
            q_samples_list, log_probs_samples = model.sample_and_log_prob(sample_size, context)
            for q_samples, lamda, log_probs in zip(q_samples_list, lambda_list, log_probs_samples):
                log_posteriors = vectorized_log_prior_unnormalized(q_samples, d, lamda, variance)
                loss = log_probs - log_posteriors
                view_q_samples_list.extend(q_samples.detach())
                view_lambda_list = np.append(view_lambda_list, [lamda.detach()] * sample_size)
                view_loss_list.append(loss.detach())
                view_log_probs.extend(log_probs.detach())
            # for q_samples, lamda, probs in zip(q_samples_list, lambda_list, log_probs_samples):
            #     log_posteriors = vectorized_log_prior_unnormalized(q_samples, d, lamda, variance)
            #     loss = log_probs_samples - log_posteriors
            #     view_q_samples_list.append(q_samples.detach())
            #     view_lambda_list.append(lamda.detach())
            #     view_loss_list.append(loss.detach())
            #     view_log_probs.append(probs.detach())

    # view_lambda_list = np.concatenate(view_lambda_list, 0)
    view_lambda_sorted_idx = view_lambda_list.argsort()
    view_lambda_list_sorted = view_lambda_list
    view_loss_list_sorted = np.array(view_loss_list)
    view_q_samples_list_sorted = np.array(view_q_samples_list)
    view_log_probs_list_sorted = np.array(view_log_probs)

    return view_q_samples_list_sorted, view_lambda_list_sorted, view_loss_list_sorted, view_log_probs_list_sorted


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


def generate_regression_dataset(n_samples, n_features, n_non_zero, noise_std):
    assert n_features >= n_non_zero

    # Generate non-zero coefficients randomly
    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    coefficients[non_zero_indices] = np.random.normal(0, 1, n_non_zero)  # Random non-zero coefficients

    # Generate data matrix X from a Gaussian distribution with covariance matrix sampled from a Wishart distribution
    scale_matrix = np.eye(n_features)  # Identity matrix as the scale matrix
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    # Sample data matrix X from a multivariate Gaussian distribution with zero mean and covariance matrix
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)

    # Generate response variable y
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std**2, n_samples)  # Linear regression model with Gaussian noise

    # compute regression parameters
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y - X @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(coefficients).float()
    # return X, y, coefficients


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
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=16,
                                num_blocks=1, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def build_sum_of_sigmoid_conditional_flow_model(d):
    context_features = 64
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
    for _ in range(num_layers):
        transforms.append(InverseTransform(
            ConditionalSumOfSigmoidsTransform(features=d, hidden_features=128,
                                              context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )
        transforms.append(InverseTransform(ActNorm(features=d)))

    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=64,
                                num_blocks=2, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def build_masked_sigmoid_conditional_flow_model(d):
    context_features = 16
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
    for _ in range(num_layers):
        transforms.append(InverseTransform(
            MaskedSumOfSigmoidsTransform
            (features=d, hidden_features=64,
                                              context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )
        transforms.append(InverseTransform(ActNorm(features=d)))

    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=16,
                                num_blocks=2, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def print_original_vs_flow_learnt_parameters(d, fixed, flows, context=None):
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

    # Set the parameters
    epochs = 100
    dimension, last_zero_indices = 5, 20
    data_sample_size = 7
    noise = 1
    likelihood_variance = torch.tensor(2)
    q_sample_size = 1

    context_size = 1000
    lambda_min_exp = -1
    lambda_max_exp = 2


    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
          f"Sample Size:{data_sample_size}, noise:{noise}\n")

    X, Y, W, variance = generate_synthetic_data(dimension, last_zero_indices, data_sample_size, noise)

    cond_min = -1
    cond_max = 2
    datadim = 5
    n_samples = 7
    sigma_regr = 2.0
    X, Y, W = generate_regression_dataset(n_samples, datadim, datadim, sigma_regr)
    # print(X, Y, W)


    # ==================================================================
    # Lasso Ground Truth
    # LassoGroundTruth.lasso_ground_truth(X, Y)

    original_W = W.tolist()


    fixed_lambda_exp = torch.rand(1)
    print("Fixed Lambda exponent: ", fixed_lambda_exp)
    # ==================================================================
    # train flows for fixed lambda with unconditional version
    # flows = build_flow_model(dimension)
    # flows, losses = LassoRegression.train(flows, dimension, X, Y, variance, epochs,
    #                                       q_sample_size, fixed_lambda_exp.item())
    # view_original_vs_flow_learnt_parameters(dimension, original_W, flows)
    #

    # ==================================================================
    # train flows for fixed lambda with CNF
    flows = build_conditional_flow_model(dimension)
    flows, losses = train(flows, dimension, X, Y, variance, epochs, q_sample_size, fixed_lambda_exp)
    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)

    # ==================================================================
    # train conditional flows

    # flows = build_conditional_flow_model(dimension)
    # flows = build_masked_sigmoid_conditional_flow_model(dimension)
    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flows.to(device)

    flows, losses = train_conditional_flows(flows, dimension, X, Y, likelihood_variance, epochs, q_sample_size,
                                            context_size, lambda_min_exp, lambda_max_exp)
    print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)

    # View.plot_loss(losses)
    # View.plot_lasso_beta_vs_lambda(dimension, flows, lambda_min_exp, lambda_max_exp)
    q_samples_list_sorted, lambda_list_sorted, loss_list_sorted, log_prob_list_sorted = sample_Ws(flows, 10,
                                                                                                  dimension, 10,
                                                                                                  lambda_min_exp,
                                                                                                  lambda_max_exp,
                                                                                                  likelihood_variance, n_iter=10)

    # print(q_samples_list_sorted, lambda_list_sorted, log_prob_list_sorted)

    View.plot_flow_lasso_path_vs_ground_truth(X, Y,
                                              flows, lambda_min_exp, lambda_max_exp,
                                              likelihood_variance, 100,
                                              device)



if __name__ == "__main__":
    main()
# device = "cuda:0" if torch.cuda.is_available() else 'cpu'

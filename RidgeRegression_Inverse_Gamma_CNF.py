import os.path

import numpy as np
import scipy as sp
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import (ConditionalShiftTransform, ConditionalScaleTransform,
                                            ConditionalLUTransform, ConditionalSumOfSigmoidsTransform)
from enflows.transforms.normalization import ActNorm
from sklearn.linear_model import LinearRegression
from scipy.special import gamma

from torch import optim

import Visualizations as View

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def vectorized_log_posterior_unnormalized(Ws, d, X, Y, a_0, b_0):
    N = len(X)
    μ_0 = torch.zeros(d).to(device)
    cov_0 = torch.eye(d).to(device)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), Y)))
    a_N = a_0 + (N / 2)
    b_N = b_0 + 0.5 * (torch.matmul(Y.t(), Y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_N.t(), Λ_N), μ_N))

    log_t_unnormalized = 0
    # log_t_unnormalized = log_t_unnormalized + torch.log(gamma(a_N + d/2) / gamma(a_N))
    # log_t_unnormalized = log_t_unnormalized + a_N * torch.log(b_N)
    log_t_unnormalized = log_t_unnormalized + -0.5 * d * torch.log(torch.tensor(2 * torch.pi))
    log_t_unnormalized = log_t_unnormalized + -0.5 * torch.log(torch.det(torch.inverse(Λ_N).to(device)))

    term_4 = Ws - μ_N
    term_3 = (torch.matmul(term_4, Λ_N) * term_4).sum(dim=-1)
    log_t_unnormalized = log_t_unnormalized + -(a_N + d / 2) * torch.log(b_N + 0.5 * term_3)

    return log_t_unnormalized


def vectorized_log_posterior_lambda_inverse_gamma_unnormalized(Ws, d, X, Y, a_0, lambda_exp):
    N = len(X)
    μ_0 = torch.zeros(d).to(device)
    cov_0 = ((1 / 10**lambda_exp).unsqueeze(-2)) * torch.eye(d).unsqueeze(0).to(device)
    b_0 = torch.tensor(N).to(device)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.bmm(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), Y)).unsqueeze(-1)).squeeze(-1)
    a_N = a_0 + (N / 2)
    b_N = b_0 + 0.5 * (torch.matmul(Y.t(), Y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0).unsqueeze(-1)
                       - torch.bmm(torch.bmm(μ_N.unsqueeze(1), Λ_N), μ_N.unsqueeze(-1)).squeeze(1)).squeeze(0)

    log_t_unnormalized = 0
    # log_t_unnormalized = log_t_unnormalized + torch.log(gamma(a_N + d/2) / gamma(a_N))
    # log_t_unnormalized = log_t_unnormalized + a_N * torch.log(b_N)
    log_t_unnormalized = log_t_unnormalized + -0.5 * d * torch.log(torch.tensor(2 * torch.pi))
    log_t_unnormalized = log_t_unnormalized + -0.5 * torch.log(torch.det(torch.inverse(Λ_N).to(device)))

    term_4 = Ws - μ_N
    term_3 = (torch.matmul(term_4, Λ_N) * term_4).sum(dim=-1)
    log_t_unnormalized = log_t_unnormalized + -(a_N + d / 2) * torch.log(b_N + 0.5 * term_3)

    return log_t_unnormalized


def train_CNF(flows, d, X, Y, epochs, n, context_size=100,
              a_0_min=-1, a_0_max=2, b_0_min=-1, b_0_max=2, lr=1e-3):
    file_name = f'CNF_d{d}_n{n}_e{epochs}_a0min{a_0_min}_a0max{a_0_max}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)

    print("Starting training the flows")
    losses = []
    loss = 0
    try:
        for epoch in range(epochs):
            # print("==================================")
            optimizer.zero_grad()
            uniform_a_0 = torch.rand(context_size).to(device)
            a_0 = (uniform_a_0 * (a_0_max - a_0_min) + a_0_min).view(-1, 1)
            uniform_b_0 = torch.rand(context_size).to(device)
            b_0 = (uniform_b_0 * (b_0_max - b_0_min) + b_0_min).view(-1, 1)
            context = torch.cat((a_0, b_0), 1)

            q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=n, context=context)
            log_p = vectorized_log_posterior_unnormalized(q_samples, d, X, Y, a_0, b_0)
            loss = torch.mean(q_log_prob - log_p)
            loss.backward()

            if epoch % 10 == 0 or epoch + 1 == epochs:
                print("Loss after iteration {}: ".format(epoch), loss.tolist())
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(flows.parameters(), 1)
            optimizer.step()

    except KeyboardInterrupt:
        print("interrupted..")

    log_marg_likelihood = compute_analytical_log_marginal_likelihood(X, Y, torch.zeros(d).to(device), torch.eye(d).to(device))
    print("Analytical Log_marginal_likelihood : ", log_marg_likelihood)
    print("Learned Log_marginal_likelihood : ", loss)
    # save_model(flows, file_name)

    return flows, losses


def train_posterior_with_lambda_inverse_gamma(flows, d, X, Y, epochs, n, context_size=100,
                                              a_0_min=-1, a_0_max=2, lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
    file_name = f'CNF_d{d}_n{n}_e{epochs}_a0min{a_0_min}_a0max{a_0_max}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)

    print("Starting training the flows")
    losses = []
    try:
        for epoch in range(epochs):
            # print("==================================")
            optimizer.zero_grad()
            uniform_a_0 = torch.rand(context_size).to(device)
            a_0 = (uniform_a_0 * (a_0_max - a_0_min) + a_0_min).view(-1, 1)
            uniform_lambda = torch.rand(context_size).to(device)
            lambda_exp = (uniform_lambda * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = torch.cat((a_0, lambda_exp), 1)

            q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=n, context=context)
            log_p = vectorized_log_posterior_lambda_inverse_gamma_unnormalized(q_samples, d, X, Y, a_0, lambda_exp)
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
    model = build_sum_of_sigmoid_conditional_flow_model(dimensions)
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
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std ** 2,
                                                   n_samples)  # Linear regression model with Gaussian noise

    # compute regression parameters
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y - X @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(coefficients).float()
    # return X, y, coefficients


def build_sum_of_sigmoid_conditional_flow_model(d):
    context_features = 16
    print("Defining the flows")

    base_dist = StandardNormal(shape=[d])
    transforms = []
    num_layers = 5
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
    embedding_net = ResidualNet(in_features=2, out_features=context_features, hidden_features=64,
                                num_blocks=5, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    model = model.to(device)
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


def sample_Ws(flows, context_size, flow_sample_size, a_0_min, a_0_max, b_0_min, b_0_max):
    # b_0_min = 1
    # b_0_max = 1
    uniform_a_0 = torch.rand(context_size).to(device)
    a_0 = (uniform_a_0 * (a_0_max - a_0_min) + a_0_min).view(-1, 1)
    uniform_b_0 = torch.rand(context_size).to(device)
    b_0 = (uniform_b_0 * (b_0_max - b_0_min) + b_0_min).view(-1, 1)
    context = torch.cat((a_0, b_0), 1)
    q_samples, q_log_ps = flows.sample_and_log_prob(flow_sample_size, context=context)

    a_0_np = a_0.cpu().detach().numpy()
    b_0_np = b_0.cpu().detach().numpy()
    q_samples = q_samples.cpu().detach().numpy()
    q_log_ps = q_log_ps.cpu().detach().numpy()
    return a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps


def compute_analytical_log_marginal_likelihood(X, y, μ_0, cov_0):
    N = len(X)
    a_0 = torch.tensor(2 * N)
    b_0 = a_0
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), y)))
    a_N = a_0 + (N / 2.)
    b_N = b_0 + 0.5 * (torch.matmul(y.t(), y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_0.t(), Λ_N), μ_N))
    term_1 = 1 / (2 * torch.pi) ** (0.5 * N)
    term_2 = (Λ_0.det() / Λ_N.det()) ** 0.5
    term_3 = (b_0 ** a_0) / (b_N ** a_N)
    term_4 = (torch.exp(torch.lgamma(a_N))) / (torch.exp(torch.lgamma(a_0)))
    log_marginal_likelihood = term_1 + term_2 + term_3 + term_4
    return log_marginal_likelihood


def compute_analytical_posterior_t_distribution_parameters(X, Y, a_0, b_0):
    d = X[0].shape[0]
    N = len(X)
    μ_0 = torch.zeros(d).to(device)
    cov_0 = torch.eye(d).to(device)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    μ_N = torch.matmul(torch.inverse(Λ_N), (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), Y)))
    a_N = a_0 + (N / 2)
    b_N = b_0 + 0.5 * (torch.matmul(Y.t(), Y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_N.t(), Λ_N), μ_N))

    mean = μ_N

    # Computing the scale matrices for all pairs of (a_0, b_0)s
    term_1 = (b_N / a_N)
    term_2 = torch.inverse(Λ_N)
    term_3 = term_1.view(term_1.shape[0], *([1] * len(term_2.shape)))
    scale_matrices = term_3 * term_2

    dfs = 2 * a_N

    return mean.cpu().detach().numpy(), scale_matrices.cpu().detach().numpy(), dfs.cpu().detach().numpy()


def sample_Ws_with_lambda(flows, context_size, flow_sample_size, lambda_min_exp, lambda_max_exp):
    uniform_lambdas = torch.rand(context_size).to(device)
    lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
    q_samples, log_ps = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
    _, lambda_sort_order = lambdas_exp.sort(0)

    # Reshaping the lambda_sort_order
    lambda_sort_order = lambda_sort_order.squeeze()

    lambda_exp_sorted = lambdas_exp[lambda_sort_order]
    lambda_sorted = 10 ** lambda_exp_sorted.squeeze().cpu().detach().numpy()
    q_samples_sorted = q_samples[lambda_sort_order].cpu().detach().numpy()
    return lambda_sorted, q_samples_sorted


def main():
    # Set the parameters
    epochs = 1000
    dimension, last_zero_indices = 3, 20
    data_sample_size = 3
    data_noise_sigma = 2.0
    q_sample_size = 1
    context_size = 1000
    a_0_min = 0
    a_0_max = 10
    b_0_min = 0
    b_0_max = 10
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, last_zero_indices:{last_zero_indices}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}\n")

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

    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows, losses = train_CNF(flows, dimension, X_torch, Y_torch, epochs,
                              q_sample_size, context_size, a_0_min, a_0_max, b_0_min, b_0_max,
                              learning_rate)

    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    # View.plot_loss(losses)
    # a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150, a_0_min, a_0_max, b_0_min, b_0_max)
    # mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, b_0)
    # View.plot_flow_ridge_inverse_gamma_parameters_1(dimension, mean, scale_matrices, dfs, a_0_np, b_0_np, q_samples,
    #                                               q_log_ps)
    # View.plot_flow_ridge_inverse_gamma_parameters(dimension, mean, scale_matrices, dfs, a_0_np, b_0_np, q_samples, q_log_ps)


    b_0_min = 0
    b_0_max = 0
    a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150, a_0_min, a_0_max, b_0_min, b_0_max)
    mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, b_0)
    View.plot_t_distributions_for_each_invergamma_parameter_pairs(dimension, mean, scale_matrices, dfs, a_0, b_0, a_0_np, b_0_np, q_samples,
                                                    q_log_ps, flows,"b_0")
    #
    # b_0_min = 1
    # b_0_max = 1
    # a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150, a_0_min, a_0_max, b_0_min, b_0_max)
    # mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, b_0)
    # View.plot_t_distributions_for_each_invergamma_parameter_pairs(dimension, mean, scale_matrices, dfs, a_0, b_0, a_0_np, b_0_np, q_samples,
    #                                                 q_log_ps, flows, "b_1")
    #
    # b_0_min = 2
    # b_0_max = 2
    # a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150, a_0_min, a_0_max, b_0_min, b_0_max)
    # mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, b_0)
    # View.plot_t_distributions_for_each_invergamma_parameter_pairs(dimension, mean, scale_matrices, dfs, a_0, b_0, a_0_np, b_0_np, q_samples,
    #                                                 q_log_ps, flows, "b_2")
    #
    # b_0_min = 5
    # b_0_max = 5
    # a_0, b_0, a_0_np, b_0_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150, a_0_min, a_0_max, b_0_min, b_0_max)
    # mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, b_0)
    # View.plot_t_distributions_for_each_invergamma_parameter_pairs(dimension, mean, scale_matrices, dfs, a_0, b_0, a_0_np, b_0_np, q_samples,
    #                                                 q_log_ps, flows, "b_5")


    # ##########################  Train posterior on a_0 and lambda ##########################
    # lambda_min_exp = -3
    # lambda_max_exp = 4
    # flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    # flows, losses = train_posterior_with_lambda_inverse_gamma(flows, dimension, X_torch, Y_torch, epochs,
    #                                                           q_sample_size, context_size, a_0_min, a_0_max, lambda_min_exp,
    #                                                           lambda_max_exp,
    #                                                           learning_rate)
    # a_0, lambda_exp, a_0_np, lambda_exp_np, q_samples, q_log_ps = sample_Ws(flows, 100, 150,
    #                                                                         a_0_min, a_0_max, lambda_min_exp, lambda_max_exp)
    # lambda_exp = 10**lambda_exp
    # # mean, scale_matrices, dfs = compute_analytical_posterior_t_distribution_parameters(X_torch, Y_torch, a_0, lambda_exp)
    # # View.plot_flow_ridge_inverse_gamma_parameters_1(dimension, mean, scale_matrices, dfs, a_0_np, lambda_exp_np, q_samples,
    # #                                               q_log_ps)
    #
    # _, lambda_sort_order = lambda_exp.sort(0)
    # lambda_exp_sorted = lambda_exp[lambda_sort_order]
    # lambda_sorted = lambda_exp_sorted.squeeze().cpu().detach().numpy()
    # q_samples_sorted = q_samples[lambda_sort_order].squeeze(1)
    # View.plot_flow_ridge_path_vs_ground_truth(X, Y, lambda_sorted, q_samples_sorted, 1, "lambda_")


if __name__ == "__main__":
    main()

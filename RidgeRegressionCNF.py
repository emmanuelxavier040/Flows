"""
This is an example of posterior estimation using flows by minimizing the KL divergence with synthetic data.
We use a Multi-variate normal distribution to generate X. We choose a fixed parameter W which can be used for a linear
transformation of X to Y. We can add some noise to the observations finally giving Y = XW + noise. Our task is to infer
about the posterior P(W | X,Y). We use linear flows to compute the KL-Divergence(q(W) || P*(W | X,Y)). Here P* is the
un-normalized posterior which is equivalent to the un-normalized Gaussian likelihood * Gaussian prior.
We can compute P* since I have X, Y and W (for W, we can easily sample from flow). After training, flows should have
learned the distribution of W and samples from it should resemble the fixed W which we used to transform X to Y.
"""
import math
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

import Evaluation
import Utilities
import Visualizations as View
import LassoRegressionCNF


# torch.manual_seed(11)
# np.random.seed(10)
torch.manual_seed(15)
np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def vectorized_log_likelihood_unnormalized(Ws, X, Y, likelihood_sigma):
    variance = torch.tensor(likelihood_sigma) ** 2
    Ws_reshaped = Ws.unsqueeze(-1)
    XWs = torch.matmul(X, Ws_reshaped)
    XWs = XWs.squeeze()
    squared_errors = (Y - XWs) ** 2
    n = Y.shape[0]
    term1 = -0.5 * n * torch.log(2 * torch.pi * variance)
    term2 = -0.5 * (1 / variance) * torch.sum(squared_errors, dim=-1)
    log_likelihood = term1 + term2
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


def vectorized_log_ridge_prior_unnormalized(Ws, sigma, lambdas_exp, d):
    variance = torch.tensor(sigma) ** 2
    lambdas_list = 10 ** lambdas_exp
    squared_weights = (Ws * Ws).sum(dim=2)
    term_1 = -0.5 * d * (torch.log(2 * torch.pi * variance) - torch.log(lambdas_list))
    term_2 = lambdas_list * (-0.5 * (1 / variance) * squared_weights)
    log_prior = term_1 + term_2
    return log_prior


def vectorized_log_posterior_unnormalized(q_samples, d, X, Y, lambdas_exp, likelihood_sigma):
    # proportional to p(Samples|q) * p(q)
    log_likelihood = vectorized_log_likelihood_unnormalized(q_samples, X, Y, likelihood_sigma)
    log_prior = vectorized_log_ridge_prior_unnormalized(q_samples, likelihood_sigma, lambdas_exp, d)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def train_for_fixed_lambda(flows, d, X, Y, variance, epochs, n, fixed_lambda_exp=torch.tensor([1.0])):
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


def train_CNF(flows, d, X, Y, X_torch, Y_torch, likelihood_sigma, epochs, n, plot_parameter_space=False, context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
    print("Starting training the flows")
    file_name = f'CNF_d{d}_n{n}_e{epochs}_lmin{lambda_min_exp}_lmax{lambda_max_exp}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    losses = []
    lambda_max_marg_likelihood = -math.inf

    T0 = 5.0
    Tn = 0.01
    cool_step_iteration = 200
    cool_num_iter = epochs // cool_step_iteration

    def cooling_function(t):
        if t < (cool_num_iter - 1):
            k = t / (cool_num_iter - 1)
            alpha = Tn / T0
            return T0 * (alpha ** k)
        else:
            return Tn

    lambdas_exp_plot = torch.linspace(lambda_min_exp, lambda_max_exp, 20)

    try:
        for epoch in range(epochs):
            t = epoch // (epochs / cool_num_iter)
            T = cooling_function(t=t)

            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = lambdas_exp
            q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=n, context=context)
            log_p = vectorized_log_posterior_unnormalized(q_samples, d, X_torch, Y_torch, lambdas_exp, likelihood_sigma)

            from scipy.special import gamma

            def beta_pdf_manual(x, alpha, beta):

                if x <= 0 or x >= 1:
                    return 0
                B_ab = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta)
                return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / B_ab

            def beta_mixture_pdf_manual(q_samples, weights, alpha_params, beta_params):
                # pdf = 0
                pdf = torch.zeros(q_samples.shape[0], q_samples.shape[1], 1)
                pdf_exp = torch.full_like(q_samples, 0)
                for weight, alpha, beta_param in zip(weights, alpha_params, beta_params):
                    cond = (q_samples > 0) & (q_samples < 1)
                    pdf_exp[~cond] = 0
                    B_ab = (gamma(alpha) * gamma(beta_param)) / gamma(alpha + beta_param)
                    pdf_exp[cond] = (q_samples[cond] ** (alpha - 1) * (1 - q_samples[cond]) ** (beta_param - 1)) / B_ab
                    pdf += weight * torch.prod(pdf_exp, dim=2, keepdim=True)
                return pdf.squeeze(1)

            weights = [0.25, 0.25, 0.25, 0.25]  # Mixing coefficients, must sum to 1
            alpha_params = [2, 5, 18, 15]  # Alpha parameters for the 4 Beta distributions
            beta_params = [8, 3, 12, 5]  # Beta parameters for the 4 Beta distributions
            log_p = beta_mixture_pdf_manual(q_samples,  weights, alpha_params, beta_params)
            # log_p = np.array([beta_mixture_pdf_manual(q_samples, x_val, y_val, weights, alpha_params, beta_params) for x_val, y_val in
            #               zip(np.ravel(X), np.ravel(Y))])

            loss = torch.mean(q_log_prob - (log_p / T))
            loss.backward()

            if plot_parameter_space and (epoch < 10 or (epoch % 10 == 0)):
                View.plot_parameter_space_3d_wth_gt(device, flows, lambdas_exp_plot[10], X_torch, Y_torch,
                                                    likelihood_sigma, title="")
                # for lamda in lambdas_exp_plot:
                #     View.plot_parameter_space_3d(device, flows, lamda, plot_marginal=False, title="During training epoch-" + str(epoch))

            if epoch % 10 == 0 or epoch + 1 == epochs:
                if epoch % cool_step_iteration == 0:
                    print("Temperature: ", T)

                print("Loss after iteration {}: ".format(epoch), loss.tolist())
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(flows.parameters(), 1)
            optimizer.step()

            # if epoch > 0 and epoch % (epochs // 200) == 0:
            #     print("Learning Rate: ", scheduler.get_last_lr())
            #     scheduler.step()

            next_T = cooling_function((epoch + 1) // (epochs / cool_num_iter))
            if next_T < 1 <= T or (T == 1. and epoch + 1 == epochs):
                if plot_parameter_space:
                    for lamda in lambdas_exp_plot:
                        View.plot_parameter_space_3d(device, flows, lamda, plot_marginal=False, title="Posterior")

                lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Y_torch,
                                                                                      likelihood_sigma, 200, 10000,
                                                                                      lambda_min_exp, lambda_max_exp)
                solution_type = "Solution Path"
                View.plot_flow_ridge_path_vs_ground_truth(X, Y, lambdas_sorted, q_samples_sorted, 1, solution_type)

                log_marg_likelihood_means = np.mean(-losses_sorted, axis=1)
                lambda_max_marg_likelihood = lambdas_sorted[np.argmax(log_marg_likelihood_means)]

                title = "Ridge-Regression-with-CNF_at_T1"
                View.plot_log_marginal_likelihood_vs_lambda(X, Y, lambdas_sorted, losses_sorted, likelihood_sigma ** 2,
                                                            title)

    except KeyboardInterrupt:
        print("interrupted..")

    # save_model(flows, file_name)

    return flows, losses, lambda_max_marg_likelihood


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

    # min_val = torch.min(W)
    # max_val = torch.max(W)
    # W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    if l < d:
        W[-l:] = 0

    Utilities.save_text_file("original_parameter.txt", str(W))

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

    context_features = 16
    hidden_features = 64
    num_layers = 3

    context_features = 32
    hidden_features = 128
    num_layers = 10

    for _ in range(num_layers):
        transforms.append(
            InverseTransform(
                ConditionalSumOfSigmoidsTransform(
                    features=d, hidden_features=hidden_features,
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
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=hidden_features,
                                num_blocks=3, activation=torch.nn.functional.relu)
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


def compute_analytical_log_marginal_likelihood(X, y, μ_0, cov_0):
    N = len(X)
    a_0 = torch.tensor(2 * N)
    b_0 = torch.tensor(2 * N)
    Λ_0 = torch.inverse(cov_0)
    Λ_N = torch.matmul(X.t(), X) + Λ_0
    Λ_N_1 = Utilities.woodbury_matrix_conversion(Λ_0, X.t(), torch.eye(X.shape[0]), X, device)
    μ_N = torch.matmul(Λ_N_1, (torch.matmul(μ_0.t(), Λ_0) + torch.matmul(X.t(), y)))
    a_N = a_0 + (N / 2.)
    b_N = b_0 + 0.5 * (torch.matmul(y.t(), y) + torch.matmul(torch.matmul(μ_0, Λ_0), μ_0) - torch.matmul(
        torch.matmul(μ_0.t(), Λ_N), μ_N))
    term_1 = 1 / ((2 * torch.pi) ** (0.5 * N))
    term_2 = (Λ_0.det() / Λ_N.det()) ** 0.5
    term_3 = (b_0 ** a_0) / (b_N ** a_N)
    term_4 = (torch.exp(torch.lgamma(a_N))) / (torch.exp(torch.lgamma(a_0)))
    log_marginal_likelihood = term_1 + term_2 + term_3 + term_4
    return log_marginal_likelihood


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


def posterior(dimension, X, Y, X_torch, Y_torch, likelihood_sigma, epochs,
              q_sample_size, context_size, lambda_min_exp, lambda_max_exp, learning_rate, plot_parameter_space, W):
    original_W = W.tolist()
    print("Original Parameters: ", original_W)
    fixed_lambda_exp = torch.rand(1)
    print("Fixed Lambda exponent: ", fixed_lambda_exp)

    # ==================================================================
    # train conditional flows

    flows = build_sum_of_sigmoid_conditional_flow_model(dimension)
    flows.to(device)
    # if plot_parameter_space:
    #     View.plot_parameter_space_3d(device, flows, torch.tensor(0.0), plot_marginal=True, title="Before Training")
    #     View.plot_parameter_space_3d(device, flows, torch.tensor(0.0), plot_marginal=False, title="Before Training")

    flows, losses, lambda_max_likelihood = train_CNF(flows, dimension, X, Y, X_torch, Y_torch,
                                                     likelihood_sigma, epochs,
                                                     q_sample_size, plot_parameter_space,
                                                     context_size, lambda_min_exp, lambda_max_exp,
                                                     learning_rate, )

    solution_type = "MAP Solution Path with Simulated Annealing"
    print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_Ws_for_plots(flows, X_torch, Y_torch,
                                                                          likelihood_sigma, 200, 10000,
                                                                          lambda_min_exp, lambda_max_exp)
    View.plot_flow_ridge_path_vs_ground_truth(X, Y, lambdas_sorted, q_samples_sorted, 1, solution_type)

    solution_type = "MAP"
    View.plot_flow_ridge_path_vs_ground_truth_standardized_coefficients(X, Y, lambdas_sorted, q_samples_sorted,
                                                                        solution_type)
    return flows, lambda_max_likelihood


def main():
    # Set the parameters
    epochs = 20000
    dimension = 5
    data_sample_size = 56

    # Configuration for plotting 3D plots of Posterior
    dimension = 2
    data_sample_size = 50

    data_noise_sigma = 0.1
    likelihood_sigma = 2
    q_sample_size = 1
    context_size = 2
    lambda_min_exp = -4
    lambda_max_exp = 5
    learning_rate = 1e-3
    plot_parameter_space = True    # Set to true to save 3D plots in between training

    print(f"============= Parameters ============= \n"
          f"Epochs:{epochs}, Dimension:{dimension}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    X, Y, W, variance = generate_synthetic_data(dimension, dimension, data_sample_size, data_noise_sigma)
    #
    # X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    X = (X - X.mean(0)) / X.std(0)

    train_ratio = 0.8
    X_train, Y_train, X_test, Y_test = Utilities.extract_train_test_data(data_sample_size, train_ratio, X, Y)

    X_torch = X_train.to(device)
    Y_torch = Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    flows, lambda_max_marg_likelihood = posterior(dimension, X_train, Y_train, X_torch, Y_torch, likelihood_sigma, epochs,
              q_sample_size, context_size, lambda_min_exp, lambda_max_exp, learning_rate, plot_parameter_space, W)

    q_selected = Utilities.select_q_for_max_likelihood_lambda(lambda_max_marg_likelihood, flows, device)

    Utilities.save_text_file("best_parameter_Ridge.txt", str(q_selected))

    Evaluation.evaluate_model(flows, q_selected, X_torch, Y_torch, "Ridge-Regression-CNf-Training-data")
    Evaluation.evaluate_model(flows, q_selected, X_test, Y_test, "Ridge-Regression-CNf-Test-data")


    flows, lambda_max_marg_likelihood = LassoRegressionCNF.posterior(X_train, Y_train, X_torch, Y_torch, likelihood_sigma,
                                                                epochs,
                                                                q_sample_size, context_size, lambda_min_exp,
                                                                lambda_max_exp,
                                                                learning_rate, W)
    q_selected = Utilities.select_q_for_max_likelihood_lambda(lambda_max_marg_likelihood, flows, device)

    Utilities.save_text_file("best_parameter_Lasso.txt", str(q_selected))

    Evaluation.evaluate_model(flows, q_selected, X_torch, Y_torch, "Lasso-Regression-CNf-Training-data")
    Evaluation.evaluate_model(flows, q_selected, X_test, Y_test, "Lasso-Regression-CNf-Test-data")


if __name__ == "__main__":
    main()

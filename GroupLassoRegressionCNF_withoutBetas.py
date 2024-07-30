
import numpy as np
import torch
from enflows.distributions.normal import StandardNormal
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.conditional import ConditionalSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from torch import optim

import GroupLassoRegressionCNF
import LassoRegressionCNF
import Visualizations as View

from enflows.transforms.nonlinearities import Softplus, Exp

# from group_lasso import GroupLasso

torch.manual_seed(11)
np.random.seed(10)
# torch.manual_seed(15)
# np.random.seed(17)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print("Device used : ", device)


def log_posterior_unnormalized(constants, grouped_indices_list, tau_samples, X, Y, lambdas_exp):
    c_1, c_2, yTΛy, c_4_JT, XTΛX = constants
    log_posterior = c_1 + c_2 + yTΛy

    n = len(X)
    log_posterior = log_posterior + (n * -0.5) * torch.log(2 * torch.tensor(torch.pi))

    n_groups = len(grouped_indices_list)
    term_1 = -0.5 * torch.sum(torch.log(tau_samples), dim=-1)
    log_posterior = log_posterior + term_1

    max_index = max(max(sublist) for sublist in grouped_indices_list) + 1
    repeat_counts = torch.tensor([len(sublist) for sublist in grouped_indices_list]).to(device)
    values_repeated = tau_samples.repeat_interleave(repeat_counts, dim=-1)
    flat_indices = [idx for sublist in grouped_indices_list for idx in sublist]
    flat_indices_tensor = torch.tensor(flat_indices).to(device)
    batch_size, depth, _ = tau_samples.shape
    list_tensors = torch.zeros((batch_size, depth, max_index)).to(device)
    list_tensors[:, :, flat_indices_tensor] = values_repeated.to(device)
    taus_diagonal_matrices = torch.diag_embed(1 / list_tensors)

    A = (XTΛX + taus_diagonal_matrices)
    term_3 = -0.5 * torch.linalg.slogdet(A)[1]
    log_posterior = log_posterior + term_3

    A_inverse = torch.inverse(A)
    term_4 = 0.5 * torch.matmul(torch.matmul(c_4_JT, A_inverse), c_4_JT.T)
    log_posterior = log_posterior + term_4

    n_features = len(X[0])
    lambda_list = 10 ** lambdas_exp

    term_5 = (- lambda_list**2 / 2) * tau_samples.sum(dim=-1)
    log_posterior = log_posterior + term_5

    term_6 = 0.5 * (n_features + n_groups) * torch.log(0.5 * (lambda_list ** 2))
    log_posterior = log_posterior + term_6

    return log_posterior


def precompute_constants(grouped_indices_list, X, Y, likelihood_cov_matrix):
    Λ = torch.inverse(likelihood_cov_matrix).to(device)
    c_1 = -0.5 * torch.linalg.slogdet(torch.inverse(Λ))[1]
    c_2 = 0
    for group_indices in grouped_indices_list:
        c_2 = c_2 - torch.lgamma(0.5 * torch.tensor(len(group_indices) + 1))

    yTΛy = -0.5 * torch.matmul(torch.matmul(Y.T, Λ), Y)

    c_4_JT = torch.matmul(torch.matmul(Y.T, ( Λ +Λ.T ) *0.5), X).to(device)
    XTΛX = torch.matmul(torch.matmul(X.T, Λ), X).to(device)

    return c_1.to(device), c_2.to(device), yTΛy, c_4_JT, XTΛX


def train_CNF(flows, d, grouped_indices_list, X, Y, X_torch, Y_torch, likelihood_sigma, epochs, tau_sample_size, context_size=100,
              lambda_min_exp=-1, lambda_max_exp=2, lr=1e-3):
    file_name = f'CNF_d{d}_n{tau_sample_size}_e{epochs}_lmin{lambda_min_exp}_lmax{lambda_max_exp}'

    optimizer = optim.Adam(flows.parameters(), lr=lr, eps=1e-8)

    print("Starting training the flows")
    losses = []

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

    try:
        n_data_samples = len(X_torch)
        likelihood_cov_matrix = likelihood_sigma * torch.eye(n_data_samples).to(device)
        constants = precompute_constants(grouped_indices_list, X_torch, Y_torch, likelihood_cov_matrix)
        for epoch in range(epochs):
            t = epoch // (epochs / cool_num_iter)
            T = cooling_function(t=t)

            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            context = lambdas_exp
            tau_samples, tau_log_prob = flows.sample_and_log_prob(num_samples=tau_sample_size, context=context)
            log_p = log_posterior_unnormalized(constants, grouped_indices_list, tau_samples, X_torch, Y_torch, lambdas_exp)
            log_p = torch.clamp(log_p, min=-1e10, max=1e10)

            loss = torch.mean(tau_log_prob - (log_p / T))
            # loss = torch.mean(tau_log_prob - log_p)
            loss.backward()

            if epoch % 10 == 0 or epoch + 1 == epochs:
                if epoch % cool_step_iteration == 0:
                    print("Temperature: ", T)
                print("Loss after iteration {}: ".format(epoch), loss.tolist())
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(flows.parameters(), 1)
            optimizer.step()

            next_T = cooling_function((epoch + 1) // (epochs / cool_num_iter))
            if next_T < 1 <= T or (T == 1. and epoch + 1 == epochs):
                lambdas_sorted, q_samples_sorted, losses_sorted = sample_from_flow_for_plots(flows, grouped_indices_list,
                                                                                 X_torch, Y_torch,
                                                                                 likelihood_sigma, 100, 100,
                                                                                 lambda_min_exp, lambda_max_exp)
                solution_type = "Group-Lasso-Solution Path"
                View.plot_flow_group_coefficients_path_vs_ground_truth(X, Y, lambdas_sorted,
                                                                       q_samples_sorted, solution_type)

                title = "GL-Without_betas_Log_marginal_likelihood"
                View.plot_log_marginal_likelihood_vs_lambda(X, Y, lambdas_sorted, losses_sorted, likelihood_sigma ** 2,
                                                            title)

    except KeyboardInterrupt:
        print("interrupted..")

    # save_model(flows, file_name)

    return flows, losses


def generate_synthetic_data(d, grouped_indices_list, zero_weight_group_index, n, noise):
    # Define a Multivariate-Normal distribution and generate some real world samples X and Y
    print("Generating real-world samples : Sample_size:{} Dimensions:{}".format(n, d))

    num_data_samples = torch.Size([n])
    num_samples = n
    X = torch.zeros((num_samples, d))

    g1_size = len(grouped_indices_list[0])
    # g1_mean = torch.randn(num_samples, 1)
    g1_mean = torch.normal(mean=10, std=3, size=(num_samples, g1_size))
    X[:, grouped_indices_list[0]] = g1_mean + torch.normal(mean=0, std=0.1, size=(num_samples, g1_size))

    g2_size = len(grouped_indices_list[1])
    # g2_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    g2_base = torch.normal(mean=10, std=3, size=(num_samples, g2_size))
    X[:, grouped_indices_list[1]] = g2_base + torch.normal(mean=0, std=0.1, size=(num_samples, g2_size))

    g3_size = len(grouped_indices_list[2])
    # g3_base = torch.rand(num_samples, 1)
    # X[:, grouped_indices_list[2]] = g3_base + torch.rand(num_samples, g3_size)
    g3_base = torch.normal(mean=10, std=3, size=(num_samples, g3_size))
    X[:, grouped_indices_list[2]] = g3_base + torch.normal(mean=0, std=0.1, size=(num_samples, g3_size))

    ##### =============================================================================

    # g4_size = len(grouped_indices_list[3])
    # g4_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    # X[:, grouped_indices_list[3]] = g4_base + torch.normal(mean=2, std=0.5, size=(num_samples, g4_size))
    #
    # g5_size = len(grouped_indices_list[4])
    # g5_base = torch.distributions.Exponential(1).sample((num_samples, 1))
    # X[:, grouped_indices_list[4]] = g5_base + torch.normal(mean=4.5, std=0.1, size=(num_samples, g5_size))

    # W = torch.rand(d) * 20 - 10
    W = torch.randn(d)

    min_val = torch.min(W)
    max_val = torch.max(W)
    W = -1 + 2 * (W - min_val) / (max_val - min_val)

    print(W.shape)
    # W = torch.tensor([1.5, 2.4, 0.3, 0.7])
    W[grouped_indices_list[zero_weight_group_index]] = 0

    v = torch.tensor(noise ** 2)
    delta = torch.randn(num_data_samples) * v
    # delta = torch.normal(0, noise ** 2, num_data_samples)
    Y = torch.matmul(X, W) + delta
    return X, Y, W, v


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

    transforms.append(InverseTransform(Softplus(eps=1e-6)))
    # transforms.append(InverseTransform(Exp()))

    transforms = transforms[::-1]
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=64,
                                num_blocks=3, activation=torch.nn.functional.relu)
    model = Flow(transform, base_dist, embedding_net=embedding_net)
    return model


def sample_from_flow_for_plots(flows, grouped_indices_list, X, Y, likelihood_sigma, context_size, flow_sample_size, lambda_min_exp, lambda_max_exp):
    num_iter = 10
    lambdas, flow_samples_list, losses = [], [], []

    with torch.no_grad():
        n_data_samples = len(X)
        likelihood_cov_matrix = likelihood_sigma * torch.eye(n_data_samples)
        constants = precompute_constants(grouped_indices_list, X, Y, likelihood_cov_matrix)
        for _ in range(num_iter):
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            q_samples, q_log_probs = flows.sample_and_log_prob(flow_sample_size, context=lambdas_exp)
            log_p_samples = log_posterior_unnormalized(constants, grouped_indices_list, q_samples, X, Y, lambdas_exp)
            loss = q_log_probs - log_p_samples

            lambdas.append((10 ** lambdas_exp).squeeze().cpu().detach().numpy())
            flow_samples_list.append(q_samples.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

    flow_samples_list, lambdas, losses = (np.concatenate(flow_samples_list, 0),
                                       np.concatenate(lambdas, 0), np.concatenate(losses, 0))
    lambda_sort_order = lambdas.argsort()

    lambdas_sorted = lambdas[lambda_sort_order]
    q_samples_sorted = flow_samples_list[lambda_sort_order]
    losses_sorted = losses[lambda_sort_order]
    return lambdas_sorted, q_samples_sorted, losses_sorted


def posterior(X, Y, X_torch, Y_torch, likelihood_sigma, grouped_indices_list, epochs, tau_sample_size,
              context_size, lambda_min_exp, lambda_max_exp, learning_rate, W):
    n_groups = len(grouped_indices_list)
    dimension = n_groups
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

    flows, losses = train_CNF(flows, dimension, grouped_indices_list, X, Y, X_torch, Y_torch,
                              likelihood_sigma, epochs,
                              tau_sample_size,
                              context_size, lambda_min_exp, lambda_max_exp,
                              learning_rate)

    # print_original_vs_flow_learnt_parameters(dimension, original_W, flows, context=fixed_lambda_exp)
    # View.plot_loss(losses)
    # solution_type = "No_Beta_Group-Lasso-Solution Path"
    solution_type = "No_Beta_Group-Lasso-MAP"
    lambdas_sorted, q_samples_sorted, losses_sorted = sample_from_flow_for_plots(flows, grouped_indices_list,
                                                                                 X_torch, Y_torch,
                                                                                 likelihood_sigma, 100, 100,
                                                                                 lambda_min_exp, lambda_max_exp)

    View.plot_flow_group_coefficients_path_vs_ground_truth(X, Y, lambdas_sorted, q_samples_sorted, solution_type)


def generate_group_indices(grouped_indices_list):
    total_indices = len(set(index for sublist in grouped_indices_list for index in sublist))

    group_indices = [0] * total_indices

    for group_id, indices in enumerate(grouped_indices_list):
        for index in indices:
            group_indices[index] = group_id

    return group_indices


def main():
    # Set the parameters
    epochs = 10000
    dimension = 10
    grouped_indices_list = [[0, 2], [1, 3, 4], [5, 6, 7, 8, 9]]
    # grouped_indices_list = [[0, 2], [1, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15], [i for i in range(16, 100)]]
    zero_weight_group_index = 2
    data_sample_size = 10
    data_noise_sigma = 2.0
    likelihood_sigma = 2
    tau_sample_size = 1
    context_size = 100
    lambda_min_exp = -3
    lambda_max_exp = 3
    learning_rate = 1e-3

    print(f"============= Parameters ============= \n"
          f"Dimension:{dimension}, zero_weight_group_index:{zero_weight_group_index}, "
          f"Sample Size:{data_sample_size}, noise:{data_noise_sigma}, likelihood_sigma:{likelihood_sigma}\n")

    X, Y, W, variance = generate_synthetic_data(dimension, grouped_indices_list, zero_weight_group_index, data_sample_size, data_noise_sigma)

    # X, Y, W = generate_regression_dataset(data_sample_size, dimension, dimension, data_noise_sigma)
    X /= X.std(0)

    X_torch = X.to(device)
    Y_torch = Y.to(device)

    posterior(X, Y, X_torch, Y_torch, likelihood_sigma, grouped_indices_list, epochs, tau_sample_size, context_size,
              lambda_min_exp, lambda_max_exp, learning_rate, W)

    # GroupLassoRegressionCNF.posterior(X, Y, X_torch, Y_torch, likelihood_sigma, grouped_indices_list, epochs, tau_sample_size,
    #                           context_size, lambda_min_exp, lambda_max_exp, learning_rate, W)

    # group_indices = generate_group_indices(grouped_indices_list)
    # group_lasso = GroupLasso(groups=group_indices, group_reg=0.1, l1_reg=0.1, n_iter=1000, old_regularisation=True)
    # group_lasso.fit(X, Y)
    # print("Learned coefficients:\n", group_lasso.coef_)

    # LassoRegressionCNF.posterior(X, Y, X_torch, Y_torch, likelihood_sigma, epochs, tau_sample_size, context_size,
    #           lambda_min_exp, lambda_max_exp, learning_rate, W)


if __name__ == "__main__":
    main()

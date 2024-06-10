import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, multivariate_t
from sklearn.linear_model import lasso_path, enet_path, LassoCV
from itertools import cycle, islice

sns.set_theme(style="darkgrid")


def plot_loss(loss_values):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(loss_values)), y=loss_values, color='blue', marker='o', markersize=5)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("./figures/Loss_CNF.pdf")
    plt.show()


def plot_analytical_flow_distribution_on_grid(num_test, mean_pred, cov_pred, flows, ground_truth,
                                              title="Distribution"):
    print("plot_analytical_flow_distribution_on_grid")
    ranges = []
    sample_size = 50

    for i in range(num_test):
        ax_min = mean_pred[i] - 3 * torch.sqrt(cov_pred[i, i])
        ax_max = mean_pred[i] + 3 * torch.sqrt(cov_pred[i, i])
        points = torch.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = torch.meshgrid(*ranges, indexing='ij')
    rv = multivariate_normal(mean_pred, cov_pred)
    analytical_density = rv.pdf(torch.stack(grids, dim=-1))

    flow_inputs = torch.cat([grid.reshape(-1, 1) for grid in grids], dim=1)
    flow_density = flows.log_prob(flow_inputs)
    reshape_argument = [sample_size for _ in range(num_test)]
    flow_density_reshaped = flow_density.reshape(*reshape_argument).exp()

    analytical_marginals = []
    flow_marginals = []
    for i in range(num_test):
        axes_to_sum = tuple(j for j in range(num_test) if j != i)
        analytical_marginal = analytical_density.sum(axis=axes_to_sum)
        flow_marginal = flow_density_reshaped.sum(axis=axes_to_sum)
        analytical_marginals.append(analytical_marginal)
        flow_marginals.append(flow_marginal)

    fig, axs = plt.subplots(1, num_test, figsize=(5 * num_test, 4))

    for i in range(num_test):
        axs[i].plot(ranges[i], analytical_marginals[i])
        axs[i].plot(ranges[i], flow_marginals[i].detach().numpy())
        axs[i].axvline(ground_truth[i], color='r', linestyle='--', label='Ground Truth')
        axs[i].set_xlabel(f"{i + 1}")
        axs[i].set_ylabel("Value")
        axs[i].legend(['Analytical', 'Flow', 'Ground Truth'])

    plt.suptitle('Analytical Vs Flow ' + title)
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_" + title + "_on_grid.pdf")
    plt.show()


def plot_analytical_flow_posterior_general_with_samples(mean_np, cov_np,
                                                        flows):
    sample_size = 10000
    flow_samples = flows.sample(sample_size).detach().numpy()
    analytical_samples = multivariate_normal.rvs(mean_np, cov_np, size=sample_size)
    analytical_samples = pd.DataFrame(analytical_samples)

    analytical_samples["type"] = "analytical"
    flow_samples = pd.DataFrame(flow_samples)
    flow_samples["type"] = "flow"
    data = pd.concat([analytical_samples, flow_samples])

    num_axes = mean_np.size
    fig, axes = plt.subplots(1, num_axes, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if ax.axison:
            sns.histplot(data=data, x=i, ax=ax, hue="type", kde=True, bins=100)
            ax.legend(['analytical', 'flow'])
            ax.set_xlabel("W_" + str(i + 1))
            ax.set_ylabel("Value")

    plt.suptitle('Analytical Vs Flow Posterior - Ridge Regression')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_normal_posterior_general_with_samples.pdf")
    plt.show()


def plot_analytical_flow_posterior_t_distribution_on_grid(dimensions, location_np, scale_matrix, df, flows):
    print("plot_analytical_flow_posterior_t_distribution_on_grid")
    ranges = []
    sample_size = 50

    for i in range(dimensions):
        ax_min = location_np[i] - 3 * torch.sqrt(scale_matrix[i, i])
        ax_max = location_np[i] + 3 * torch.sqrt(scale_matrix[i, i])
        points = torch.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = torch.meshgrid(*ranges, indexing='ij')
    rv = multivariate_t(loc=location_np, shape=scale_matrix, df=df)
    analytical_density = rv.pdf(torch.stack(grids, dim=-1))

    flow_inputs = torch.cat([grid.reshape(-1, 1) for grid in grids], dim=1)
    flow_density = flows.log_prob(flow_inputs)
    reshape_argument = [sample_size for _ in range(dimensions)]
    flow_density_reshaped = flow_density.reshape(*reshape_argument).exp()

    analytical_marginals = []
    flow_marginals = []
    for i in range(dimensions):
        axes_to_sum = tuple(j for j in range(dimensions) if j != i)
        analytical_marginal = analytical_density.sum(axis=axes_to_sum)
        flow_marginal = flow_density_reshaped.sum(axis=axes_to_sum)
        analytical_marginals.append(analytical_marginal)
        flow_marginals.append(flow_marginal)

    fig, axs = plt.subplots(1, dimensions, figsize=(5 * dimensions, 4))

    for i in range(dimensions):
        axs[i].plot(ranges[i], analytical_marginals[i])
        axs[i].plot(ranges[i], flow_marginals[i].detach().numpy())
        axs[i].set_xlabel(f"W_{i + 1}")
        axs[i].set_ylabel("Value")
        axs[i].legend(['analytical', 'flow'])

    plt.suptitle('Analytical Vs Flow Posterior - T-distribution from Inv Gamma')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_T-Posterior-General_on_grid.pdf")
    plt.show()


def plot_analytical_flow_posterior_predictive_t_distribution_on_grid(dimensions, location_np, scale_matrix, df, flows):
    print("plot_analytical_flow_posterior_predictive_t_distribution_on_grid")
    ranges = []
    sample_size = 50

    for i in range(dimensions):
        ax_min = location_np[i] - 3 * torch.sqrt(scale_matrix[i, i])
        ax_max = location_np[i] + 3 * torch.sqrt(scale_matrix[i, i])
        points = torch.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = torch.meshgrid(*ranges, indexing='ij')
    rv = multivariate_t(loc=location_np, shape=scale_matrix, df=df)
    analytical_density = rv.pdf(torch.stack(grids, dim=-1))

    flow_inputs = torch.cat([grid.reshape(-1, 1) for grid in grids], dim=1)
    flow_density = flows.log_prob(flow_inputs)
    reshape_argument = [sample_size for _ in range(dimensions)]
    flow_density_reshaped = flow_density.reshape(*reshape_argument).exp()

    analytical_marginals = []
    flow_marginals = []
    for i in range(dimensions):
        axes_to_sum = tuple(j for j in range(dimensions) if j != i)
        analytical_marginal = analytical_density.sum(axis=axes_to_sum)
        flow_marginal = flow_density_reshaped.sum(axis=axes_to_sum)
        analytical_marginals.append(analytical_marginal)
        flow_marginals.append(flow_marginal)

    fig, axs = plt.subplots(1, dimensions, figsize=(5 * dimensions, 4))

    for i in range(dimensions):
        axs[i].plot(ranges[i], analytical_marginals[i])
        axs[i].plot(ranges[i], flow_marginals[i].detach().numpy())
        axs[i].axvline(location_np[i], color='r', linestyle='--', label='Mean')
        axs[i].set_xlabel(f"W_{i + 1}")
        axs[i].set_ylabel("Value")
        axs[i].legend(['analytical', 'flow'])

    plt.suptitle('Analytical Vs Flow Posterior Predictive - T-distribution from Inv gamma')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_posterior_predictive_t_general_on_grid.pdf")
    plt.show()


def plot_lasso_beta_vs_lambda(dimension, flows, lambda_exp_min=-1, lambda_exp_max=4):
    context_size = 100
    flow_sample_size = 100
    uniform_lambdas = torch.rand(context_size)
    lambdas_exp = (uniform_lambdas * (lambda_exp_max - lambda_exp_min) + lambda_exp_min).view(-1, 1)
    context = lambdas_exp
    q_samples, log_ps = flows.sample_and_log_prob(flow_sample_size, context)
    log_ps = torch.exp(log_ps)
    lambda_values = lambdas_exp.unsqueeze(1).expand(-1, q_samples.shape[1], lambdas_exp.shape[1])
    new_weight_vectors = torch.cat((lambda_values, q_samples), dim=2)
    log_probs = torch.reshape(log_ps, (log_ps.shape[0], log_ps.shape[1], 1))
    new_weight_vectors_with_log_probs = torch.cat((new_weight_vectors, log_probs), dim=2)
    final_tensor = new_weight_vectors_with_log_probs.view(-1, new_weight_vectors_with_log_probs.shape[-1])
    data = pd.DataFrame(final_tensor.detach().numpy())
    print(data)

    for i in range(dimension):
        sns.lineplot(x=0, y=i + 1, data=data, label='W_' + str(i))

    plt.title('Beta Coefficients Vs Lambda - Lasso Regression')
    plt.xlabel(r'$\lambda$', fontsize=15)
    plt.xscale('log')

    plt.ylabel(r'$\beta$', fontsize=15)
    plt.tight_layout()
    plt.legend()
    plt.savefig("./figures/CNF_lasso_beta_vs_lambda.pdf")
    plt.show()


def plot_flow_ridge_path_vs_ground_truth(X, Y, lambda_sorted, q_samples_sorted, variance=1):
    dimension = X.shape[1]
    n = Y.shape[0]

    print("Computing regularization path using Sklearn lasso...")
    alphas_lasso, coefs_lasso, _ = enet_path(X, Y, alphas=lambda_sorted, l1_ratio=0)
    alphas_lasso = alphas_lasso * n / variance

    x = lambda_sorted
    y = q_samples_sorted

    p95 = np.quantile(y, 0.95, axis=1)
    p5 = np.quantile(y, 0.05, axis=1)
    means = np.mean(y, axis=1)

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    colors = list(islice(cycle(colors), dimension))
    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        plt.plot(alphas_lasso, coefs_lasso[index], label='G_' + str(index), color=colors[index], linestyle='--')
        p95_np = p95[:, index]
        p5_np = p5[:, index]
        mean_np = means[:, index]
        plt.plot(x, mean_np, color=colors[index], label='W_' + str(index))
        plt.fill_between(x, p5_np, p95_np, color=colors[index], alpha=0.1)

    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Coefficients')
    plt.title('Ridge Regression with CNF')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/GroundTruth_vs_Flow_Ridge_paths.pdf")
    plt.show()


def plot_flow_ridge_inverse_gamma_parameters(a_0, b_0, q_log_ps):
    print("Plot a_0 and b_0 of T-distribution against the log_probabilities of Beta samples")
    q_log_ps = np.exp(q_log_ps)
    q_samples_mean = np.mean(q_log_ps, axis=1)

    a_0_flat = a_0.flatten()
    b_0_flat = b_0.flatten()

    grid_x, grid_y = np.mgrid[a_0_flat.min():a_0_flat.max():100j, b_0_flat.min():b_0_flat.max():100j]

    grid_z = griddata((a_0_flat, b_0_flat), q_samples_mean, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')
    plt.colorbar(contour, label='Mean of q_samples')
    plt.xlabel('a_0')
    plt.ylabel('b_0')
    plt.title('Contour plot of a_0 and b_0 coloured by mean of q_samples')
    plt.savefig("./figures/Ridge_CNF_Inv_Gamma_parameters.pdf")
    plt.show()
    return


#
# def plot_flow_ridge_path_vs_ground_truth(X, Y, flows, lambda_min_exp=-1, lambda_max_exp=4,
#                                          context_size=100, flow_sample_size=150, device="cpu"):
#     dimension = X.shape[1]
#     n = Y.shape[0]
#     uniform_lambdas = np.random.rand(context_size)
#     lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp)
#     lambdas_exp = 10 ** lambdas_exp
#
#     print("Computing ground truth regularization path using the Ridge Regression ...")
#     alphas_lasso, coefs_lasso, _ = enet_path(X, Y, alphas=lambdas_exp, l1_ratio=0)
#     alphas_lasso = alphas_lasso * n
#     neg_log_alphas_lasso = np.log10(alphas_lasso)
#
#     uniform_lambdas = torch.rand(context_size).to(device)
#     lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
#     context = lambdas_exp
#     q_samples, log_ps = flows.sample_and_log_prob(flow_sample_size, context)
#     lambda_sorted, sorted_indices = lambdas_exp.sort(0)
#     sorted_indices = sorted_indices.squeeze()
#     lambda_sorted = lambdas_exp[sorted_indices].squeeze()
#     q_sorted = q_samples[sorted_indices]
#
#     x_np = lambda_sorted.cpu().detach().numpy()
#     y = q_sorted.cpu().detach().numpy()
#     p95 = np.quantile(y, 0.95, axis=1)
#     p5 = np.quantile(y, 0.05, axis=1)
#     means = np.mean(y, axis=1)
#     d = dimension
#     p95 = np.stack(tuple(p95))
#     p5 = np.stack(tuple(p5))
#     means = np.stack(tuple(means))
#
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
#
#     plt.figure(figsize=(10, 5))
#
#     for index in range(d):
#         p95_np = p95[:, index]
#         p5_np = p5[:, index]
#         mean_np = means[:, index]
#         plt.plot(neg_log_alphas_lasso, coefs_lasso[index], label='G_' + str(index), color=colors[index], linestyle='--')
#         plt.plot(x_np, mean_np, color=colors[index], label='W_' + str(index))
#         plt.fill_between(x_np, p5_np, p95_np, color=colors[index], alpha=0.1)
#
#     plt.xlabel('Lambda')
#     plt.ylabel('Coefficients')
#     plt.title('Ridge Regression with CNF')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("./figures/GroundTruth_Ridge_vs_flow_ridge_paths.pdf")
#     plt.show()

def plot_lasso_path_variance(lambdas_sorted, q_samples_sorted):
    num_subplots = 2
    min_lambda = np.round(lambdas_sorted[0], 3)
    max_lambda = np.round(lambdas_sorted[-1], 3)
    fig, axs = plt.subplots(1, num_subplots, figsize=(15, 8))
    axs[0].boxplot(x=q_samples_sorted[0])
    axs[0].set_xlabel(f"$\lambda$ = " + str(min_lambda))
    axs[0].set_ylabel("Coefficient")
    axs[1].boxplot(x=q_samples_sorted[-1])
    axs[1].set_xlabel(f"$\lambda$ = " + str(max_lambda))
    plt.suptitle('Coefficient Vs Max and Min Lambda- Lasso Regression')
    plt.savefig("./figures/Coefficient_Vs_Max_Min_Lambda_Lasso_CNF.pdf")
    plt.tight_layout()
    plt.show()
    return


def plot_flow_lasso_path_vs_ground_truth(X, Y, lambda_sorted, q_samples_sorted, variance=1):
    dimension = X.shape[1]
    n = Y.shape[0]

    print("Computing regularization path using Sklearn lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, Y, alphas=lambda_sorted)
    alphas_lasso = alphas_lasso * n / variance

    x = lambda_sorted
    y = q_samples_sorted

    p95 = np.quantile(y, 0.95, axis=1)
    p5 = np.quantile(y, 0.05, axis=1)
    means = np.mean(y, axis=1)

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    colors = list(islice(cycle(colors), dimension))

    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        plt.plot(alphas_lasso, coefs_lasso[index], label='G_' + str(index), color=colors[index], linestyle='--')
        p95_np = p95[:, index]
        p5_np = p5[:, index]
        mean_np = means[:, index]
        plt.plot(x, mean_np, color=colors[index], label='W_' + str(index))
        plt.fill_between(x, p5_np, p95_np, color=colors[index], alpha=0.1)

    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Coefficients')
    plt.title('Lasso Regression with CNF')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/GroundTruth_lasso_vs_flow_lasso_paths.pdf")
    plt.show()


def plot_log_marginal_likelihood_vs_lambda(X, Y, lambda_sorted, losses_sorted, variance,
                                           title="Distribution"):
    # print("Lambda sorted : ", lambda_sorted)
    n = Y.shape[0]
    y = -losses_sorted
    p95 = np.quantile(y, 0.95, axis=1)
    p5 = np.quantile(y, 0.05, axis=1)
    means = np.mean(y, axis=1)
    lambda_max_likelihood = lambda_sorted[np.argmax(means)]
    print("Best learnt lambda: ", lambda_max_likelihood)
    print("Max Log marginal likelihood: ", np.max(means))

    alphas = lambda_sorted
    alpha_lasso_cv = LassoCV(alphas=alphas, cv=5).fit(X, Y).alpha_
    print("Lasso CV lambda: ", alpha_lasso_cv)
    # alpha_lasso_cv = 10 ** (np.log10(alpha_lasso_cv)*n/variance)
    best_lambda_lasso_cv = alpha_lasso_cv * n / variance
    print("Lasso CV lambda rescaled: ", best_lambda_lasso_cv)

    plt.figure(figsize=(10, 5))
    plt.plot(lambda_sorted, means)
    # plt.fill_between(lambda_sorted, p95, p5, color='g', alpha=0.3)

    plt.title('Log Marginal Likelihood vs Lambda')
    # plt.xlabel(r'$\lambda$', fontsize=16)
    # plt.ylabel(r'$\log( p(X,Y \mid \lambda))$', fontsize=16)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Log marginal likelihood')
    plt.xscale('log')

    plt.vlines(best_lambda_lasso_cv, p5.min(), means.max(), label='$\lambda* LassoCV$', colors='r', linestyles='dashed')
    plt.vlines(lambda_max_likelihood, p5.min(), means.max(), label='$\lambda* Flow$', colors='b', linestyles='dashed')

    plt.legend()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("./figures/Log_Marginal_Likelihood-"+title+".pdf")
    plt.show()


def backup_bug_hunt_plot_flow_lasso_path_vs_ground_truth(dimension, X, y, flows, lambda_min_exp=-1, lambda_max_exp=4,
                                                         variance=1, context_size=500, device="cpu", show_plot=True,
                                                         print_index=0):
    flow_sample_size = 150
    n = y.shape[0]
    uniform_lambdas = np.random.rand(context_size)
    lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp)
    lambdas_exp = 10 ** lambdas_exp

    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=lambdas_exp)
    alphas_lasso = alphas_lasso * n / variance.cpu().detach().numpy()

    uniform_lambdas = torch.rand(context_size).to(device)
    # uniform_lambdas = torch.tensor(uniform_lambdas, dtype=torch.float32).to(device)
    lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
    context = lambdas_exp
    q_samples, log_ps = flows.sample_and_log_prob(flow_sample_size, context)
    log_ps = torch.exp(log_ps)

    # print(q_samples)

    lambda_values = lambdas_exp.unsqueeze(1).expand(-1, q_samples.shape[1], lambdas_exp.shape[1])
    new_weight_vectors = torch.cat((lambda_values, q_samples), dim=2)
    log_probs = torch.reshape(log_ps, (log_ps.shape[0], log_ps.shape[1], 1))
    new_weight_vectors_with_log_probs = torch.cat((new_weight_vectors, log_probs), dim=2)
    final_tensor = new_weight_vectors_with_log_probs.view(-1, new_weight_vectors_with_log_probs.shape[-1])
    data = pd.DataFrame(final_tensor.cpu().detach().numpy())
    data = data.sort_values(by=0, ascending=True)
    if show_plot:
        print(data)
    # Display results
    plt.figure(1)

    colors = [
        "#00008B",  # Dark Blue
        "#0000CD",  # Medium Blue
        "#0000FF",  # Blue
        "#4169E1",  # Royal Blue
        "#6495ED",  # Cornflower Blue
        "#87CEEB",  # Sky Blue
        "#B0E2FF",  # Light Blue
        "#EDF5FF"  # Alice Blue
    ]

    trained_colors = [
        "#8B0000",  # Dark Red
        "#DC143C",  # Crimson
        "#FF0000",  # Red
        "#FF6347",  # Tomato
        "#FFA07A",  # Light Salmon
        "#FFC0CB"  # Pink
    ]

    colors = list(islice(cycle(colors), dimension))
    trained_colors = list(islice(cycle(trained_colors), dimension))

    # neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_lasso = np.log10(alphas_lasso)
    # neg_log_alphas_lasso = alphas_lasso

    lines1 = []
    lines2 = []
    for index in range(dimension):
        plt.plot(neg_log_alphas_lasso, coefs_lasso[index], label='G_' + str(index), color=colors[index], linestyle='--')
        lines1.append(str(index))
        sns.lineplot(x=0, y=index + 1, data=data, label='W_' + str(index), color=trained_colors[index])
        lines2.append('W_' + str(index))
        # break

    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("coefficients")
    plt.title("Lasso")
    plt.axis("tight")

    if show_plot:
        plt.savefig("./figures/GroundTruth_lasso_vs_flow_lasso_paths.pdf")
        plt.show()
    else:
        plt.savefig("./figures/Result_" + str(print_index) + ".pdf")

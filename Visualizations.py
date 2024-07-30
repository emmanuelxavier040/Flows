import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata

import numpy as np

from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, multivariate_t
from sklearn.linear_model import lasso_path, enet_path, LassoCV, RidgeCV
from itertools import cycle, islice

sns.set_theme(style="darkgrid")

import rpy2.robjects as robjects
import pandas as pd
import numpy as np
import torch
from rpy2.robjects import pandas2ri

pandas2ri.activate()
r = robjects.r
r['source']('GLasso.R')

glasso_path = robjects.globalenv['GLassoPath']
cv_gglasso = robjects.globalenv['CVLasso']


def plot_loss(loss_values):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(loss_values)), y=loss_values, color='blue', marker='o', markersize=5)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("./figures/Loss_CNF.pdf")
    plt.show()


def plot_analytical_flow_distribution_on_grid(num_elements, mean_pred, cov_pred, flows, ground_truth,
                                              title="Distribution"):
    print("plot_analytical_flow_distribution_on_grid")
    ranges = []
    sample_size = 50

    for i in range(num_elements):
        ax_min = mean_pred[i] - 3 * torch.sqrt(cov_pred[i, i])
        ax_max = mean_pred[i] + 3 * torch.sqrt(cov_pred[i, i])
        points = torch.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = torch.meshgrid(*ranges, indexing='ij')
    rv = multivariate_normal(mean_pred, cov_pred)
    analytical_density = rv.pdf(torch.stack(grids, dim=-1))

    flow_inputs = torch.cat([grid.reshape(-1, 1) for grid in grids], dim=1)
    flow_density = flows.log_prob(flow_inputs)
    reshape_argument = [sample_size for _ in range(num_elements)]
    flow_density_reshaped = flow_density.reshape(*reshape_argument).exp()

    analytical_marginals = []
    flow_marginals = []
    for i in range(num_elements):
        axes_to_sum = tuple(j for j in range(num_elements) if j != i)
        analytical_marginal = analytical_density.sum(axis=axes_to_sum)
        flow_marginal = flow_density_reshaped.sum(axis=axes_to_sum)
        analytical_marginals.append(analytical_marginal)
        flow_marginals.append(flow_marginal)

    fig, axs = plt.subplots(1, num_elements, figsize=(5 * num_elements, 4))

    for i in range(num_elements):
        axs[i].plot(ranges[i], analytical_marginals[i])
        axs[i].plot(ranges[i], flow_marginals[i].detach().numpy())
        # axs[i].axvline(ground_truth[i], color='r', linestyle='--', label='Ground Truth')
        if "predictive" not in title.lower():
            x_label = r'$\beta$' + str(i)
        else:
            x_label = r'$y' + str(i) + '_{pred}$'
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel("Value")
        axs[i].legend(['$Analytical$', '$Flow$', '$Ground\: Truth$'])

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
    fig, axes = plt.subplots(1, num_axes, figsize=(15, 4))

    for i, ax in enumerate(axes.flat):
        if ax.axison:
            sns.histplot(data=data, x=i, ax=ax, hue="type", kde=True, bins=100)
            ax.legend(['$Analytical$', '$Flow$'])
            ax.set_xlabel(r'$\beta$' + str(i))
            ax.set_ylabel("Number of Samples")

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
        axs[i].set_xlabel(r'$\beta$' + str(i))
        axs[i].set_ylabel("Value")
        axs[i].legend(['$Analytical$', '$Flow$'])

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
        # axs[i].axvline(location_np[i], color='r', linestyle='--', label='Mean')
        axs[i].set_xlabel(r'$y$' + str(i))
        axs[i].set_ylabel("Value")
        axs[i].legend(['$Analytical$', '$Flow$'])

    plt.suptitle('Analytical Vs Flow Posterior Predictive - T-distribution from Inv gamma')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_posterior_predictive_t_general_on_grid.pdf")
    plt.show()


def plot_flow_ridge_inverse_gamma_parameters_1(dimension, mean, scale_matrices, dfs, a_0, b_0, q_samples, q_log_ps,
                                               title=""):
    print("Plot a_0 and b_0 of T-distribution against the Variance of Beta samples")

    markers = ['o', 's', '^', 'x', '+', 'v']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    A = a_0.squeeze()
    A_sort_idx = A.argsort()
    B = b_0.squeeze()
    B_sort_idx = B.argsort()
    variance_list = np.var(q_samples, axis=1)

    legend_handles1 = []
    legend_handles2 = []

    for i in range(dimension):
        C_gr_truth = scale_matrices[:, i, i]
        C_flow = variance_list[:, i]

        l11 = ax1.plot(A[A_sort_idx], C_gr_truth[A_sort_idx], markersize=4, marker=markers[i], linestyle='-',
                       label=r'$\sigma^2_{\beta' + str(i) + '}$')
        l12 = ax1.plot(A[A_sort_idx], C_flow[A_sort_idx], markersize=4, marker=markers[i], linestyle='--',
                       label=r'$\sigma^2_{\beta' + str(i) + '}$')
        legend_handles1.extend([l11, l12])

        l21 = ax2.plot(B[B_sort_idx], C_gr_truth[B_sort_idx], markersize=4, marker=markers[i], linestyle='-',
                       label=r'$\sigma^2_{\beta' + str(i) + '}$')
        l22 = ax2.plot(B[B_sort_idx], C_flow[B_sort_idx], markersize=4, marker=markers[i], linestyle='--',
                       label=r'$\sigma^2_{\beta' + str(i) + '}$')
        legend_handles2.extend([l21, l22])

    legend_handles1_flat = [handle for sublist in legend_handles1 for handle in sublist]
    legend_handles2_flat = [handle for sublist in legend_handles2 for handle in sublist]
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label=r'$Ground\: Truth$'),
        Line2D([0], [0], color='black', linestyle='--', label=r'$Flow$')
    ]
    legend_handles1_flat.extend(legend_elements)
    legend_handles2_flat.extend(legend_elements)

    ax1.legend(handles=legend_handles1_flat, loc='upper right')
    ax2.legend(handles=legend_handles2_flat, loc='upper right')
    ax1.set_xlabel(r'$a_0$')
    ax1.set_ylabel('Variance')
    ax1.set_title(r'Variance of $\beta$ compared to $a_0$')
    ax2.set_xlabel(r'$b_0$')
    ax2.set_ylabel('Variance')
    ax2.set_title(r'Variance of $\beta$ compared to $b_0$')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/Ridge_CNF_Inv_Gamma_parameters-" + title + ".pdf")
    plt.show()
    return


def plot_flow_ridge_inverse_gamma_parameters(dimension, mean, scale_matrices, dfs, a_0, b_0, q_samples, q_log_ps):
    print("Plot a_0 and b_0 of T-distribution against the Variance of Beta samples")

    markers = ['o', 's', '^', 'x', '+', 'v']
    A = a_0.squeeze()
    B = b_0.squeeze()
    variance_list = np.var(q_samples, axis=1)
    C_gr_truth = scale_matrices[:, 0, 0]
    C_flow = variance_list[:, 0]
    C = np.exp(q_log_ps.mean(axis=1))

    # Assuming A, B, and C are already defined as numpy arrays

    # Create grid data for the contour plot
    grid_x, grid_y = np.mgrid[min(A):max(A):100j, min(B):max(B):100j]
    grid_z = griddata((A, B), C, (grid_x, grid_y), method='cubic')

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.7)

    # Add contour plot on the XY plane
    ax.contourf(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap='viridis', alpha=0.5)

    # Add contour plot on the surface
    ax.contour3D(grid_x, grid_y, grid_z, 50, cmap='viridis')

    # Set labels
    ax.set_xlabel('A (X-axis)')
    ax.set_ylabel('B (Y-axis)')
    ax.set_zlabel('C (Z-axis)')
    ax.set_zlim(0, np.max(C))

    # Show plot
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

    return


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
    plt.suptitle('Coefficient Vs Max and Min Lambda - Lasso Regression')
    plt.savefig("./figures/Coefficient_Vs_Max_Min_Lambda_Lasso_CNF.pdf")
    plt.tight_layout()
    plt.show()
    return


def convert_group_indices_for_gglasso(grouped_indices_list):
    max_index = max(max(sublist) for sublist in grouped_indices_list) + 1
    group = [-1] * max_index
    for group_index, sublist in enumerate(grouped_indices_list):
        for index in sublist:
            group[index] = group_index + 1
    return np.array(group)


# ============================= Solution Paths plots =============================


def plot_flow_ridge_path_vs_ground_truth(X, Y, lambda_sorted, q_samples_sorted, variance=1, solution_type="Solution Path"):
    n = Y.shape[0]
    x_flow = lambda_sorted
    y_flow = q_samples_sorted

    alphas_ridge, coefs_ridge, _ = enet_path(X, Y, alphas=lambda_sorted, l1_ratio=0)
    alphas_ridge = alphas_ridge * n / variance
    x_enet_path = alphas_ridge
    y_enet_path = coefs_ridge

    plot_title = 'Ridge Regression with CNF - ' + solution_type

    plot_flow_path_vs_ground_truth(x_enet_path, y_enet_path, x_flow, y_flow, plot_title, beta_group_map={}, group_lasso=False)


def plot_flow_lasso_path_vs_ground_truth(X, Y, lambda_sorted, q_samples_sorted, variance=1, solution_type="Solution Path"):
    n = Y.shape[0]
    x_flow = lambda_sorted
    y_flow = q_samples_sorted

    alphas_lasso, coefs_lasso, _ = lasso_path(X, Y, alphas=lambda_sorted)
    alphas_lasso = alphas_lasso * n / variance
    x_lasso_path = alphas_lasso
    y_lasso_path = coefs_lasso

    plot_title = 'Lasso Regression with CNF - ' + solution_type

    plot_flow_path_vs_ground_truth(x_lasso_path, y_lasso_path, x_flow, y_flow, plot_title, beta_group_map={}, group_lasso=False)


def plot_flow_group_lasso_path_vs_ground_truth(X, Y, grouped_indices_list, lambda_sorted, q_samples_sorted, solution_type):

    lambda_flow = lambda_sorted
    coeff_flow = q_samples_sorted

    group = convert_group_indices_for_gglasso(grouped_indices_list)
    W_glasso = glasso_path(X, Y, group, np.array(lambda_sorted))
    lambda_glasso = np.array(W_glasso[4])
    coeff_glasso = np.array(W_glasso[1])

    beta_group_map = {}
    for group_index, group in enumerate(grouped_indices_list):
        for beta_index in group:
            beta_group_map[beta_index] = group_index

    plot_title = 'Group Lasso Regression with CNF - ' + solution_type

    plot_flow_path_vs_ground_truth(lambda_glasso, coeff_glasso, lambda_flow, coeff_flow, plot_title, beta_group_map, group_lasso=True)


def plot_flow_path_vs_ground_truth(lambda_gt, coeff_gt, lambda_flow, coeff_flow, plot_title, beta_group_map, group_lasso):
    dimension = coeff_flow.shape[-1]

    p95 = np.quantile(coeff_flow, 0.95, axis=1)
    p5 = np.quantile(coeff_flow, 0.05, axis=1)
    y_flow_mean = np.mean(coeff_flow, axis=1)

    colors = ['blue', 'green', 'red', 'brown', 'magenta', 'yellow', 'black', 'orange', 'purple', 'cyan']
    colors = list(islice(cycle(colors), dimension))

    # my_linestyles = [
    #     'solid',
    #     (0, (1, 1)),
    #     'dashed']
    # my_linestyles = list(islice(cycle(my_linestyles), dimension))

    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        color = colors[beta_group_map[index]] if group_lasso else colors[index]
        p95_np = p95[:, index]
        p5_np = p5[:, index]
        y_value = y_flow_mean[:, index]
        plt.plot(lambda_flow, y_value, color=color, label=r'$\beta$' + str(index))
        plt.fill_between(lambda_flow, p5_np, p95_np, color=color, alpha=0.1)

        y_value = coeff_gt[index]
        plt.plot(lambda_gt, y_value, linestyle="dashed", color=color, label=r'$\beta$' + str(index))

    plt.xlabel(r'$\lambda$')
    plt.xscale('log')
    plt.ylabel('Coefficients')
    plt.title(plot_title)
    plt.legend()
    plt.savefig("./figures/GT_vs_flow_path_" + plot_title.lower().replace(" ", "_") + ".pdf", dpi=300)
    plt.show()


# ============================= Solution Paths with Standardized Coefficient plots =============================


def plot_flow_ridge_path_vs_ground_truth_standardized_coefficients(X, Y, lambda_sorted, q_samples_sorted, solution_type):
    plot_title = "Ridge with CNF - Standardized Coefficients - " + solution_type
    alphas_lasso, coefs_lasso, _ = enet_path(X, Y, alphas=lambda_sorted, l1_ratio=0)
    ridge_coeff_estimate = coefs_lasso.T
    plot_flow_path_vs_ground_truth_standardized_coefficients(q_samples_sorted, ridge_coeff_estimate, plot_title,
                                                             beta_group_map={}, group_lasso=False)


def plot_flow_lasso_path_vs_ground_truth_standardized_coefficients(X, Y, lambda_sorted, q_samples_sorted, solution_type):
    plot_title = "Lasso with CNF - Standardized Coefficients - " + solution_type
    alphas_lasso, coefs_lasso, _ = lasso_path(X, Y, alphas=lambda_sorted)
    lasso_coeff_estimate = coefs_lasso.T
    plot_flow_path_vs_ground_truth_standardized_coefficients(q_samples_sorted, lasso_coeff_estimate, plot_title,
                                                             beta_group_map={}, group_lasso=False)


def plot_flow_group_lasso_path_vs_ground_truth_standardized_coefficients(X, Y, grouped_indices_list, lambda_sorted,
                                                                         q_samples_sorted, solution_type):
    beta_group_map = {}
    for group_index, group in enumerate(grouped_indices_list):
        for beta_index in group:
            beta_group_map[beta_index] = group_index

    group = convert_group_indices_for_gglasso(grouped_indices_list)
    W_glasso = glasso_path(X, Y, group, np.array(lambda_sorted))
    gglasso_coeff_estimate = W_glasso[1].T

    plot_title = "Group-Lasso with CNF - Standardized Coefficients - " + solution_type

    plot_flow_path_vs_ground_truth_standardized_coefficients(q_samples_sorted, gglasso_coeff_estimate, plot_title,
                                                             beta_group_map, group_lasso=True)


def plot_flow_path_vs_ground_truth_standardized_coefficients(q_samples_sorted, gt_coeffs, plot_title, beta_group_map,
                                                             group_lasso):
    dimension = q_samples_sorted.shape[-1]

    flow_coeff_estimate = np.median(q_samples_sorted, axis=1)
    flow_lambda = np.linalg.norm(flow_coeff_estimate, axis=1) / np.linalg.norm(flow_coeff_estimate, axis=1).max()
    flow_lambda_sort_order = flow_lambda.argsort()
    x_flows = flow_lambda[flow_lambda_sort_order]
    y_flows = flow_coeff_estimate[flow_lambda_sort_order]

    gt_coeff_estimate = gt_coeffs
    gt_lambdas = np.linalg.norm(gt_coeff_estimate, axis=1) / np.linalg.norm(gt_coeff_estimate, axis=1).max()
    gt_lambdas_sort_order = gt_lambdas.argsort()
    x_gt = gt_lambdas[gt_lambdas_sort_order]
    y_gt = gt_coeff_estimate[gt_lambdas_sort_order]

    colors = ['blue', 'green', 'red', 'brown', 'magenta', 'yellow', 'black', 'orange', 'purple', 'cyan']
    colors = list(islice(cycle(colors), dimension))

    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        color = colors[beta_group_map[index]] if group_lasso else colors[index]
        y_value = y_flows[:, index]
        plt.plot(x_flows, y_value, color=color, label=r'$\beta$' + str(index))
        y_value = y_gt[:, index]
        plt.plot(x_gt, y_value, linestyle="dashed", color=color, label=r'$\beta$' + str(index))

    plt.xlabel(r'$|\beta|$ / max $|\beta|$')
    plt.ylabel('Coefficients')
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(
        "./figures/GT_vs_flow_path_standardized_coeff_" + plot_title.lower().replace(" ", "_") + ".pdf",
        dpi=300)
    plt.show()

# ==============================================================================================================


def convert_q_to_group_norms(q_samples_sorted, grouped_indices_list):
    norm_array = np.empty((q_samples_sorted.shape[0], q_samples_sorted.shape[1], len(grouped_indices_list)))
    for group_index, indices in enumerate(grouped_indices_list):
        group = q_samples_sorted[:, :, indices]
        norm_array[:, :, group_index] = np.linalg.norm(group, axis=-1)
    return norm_array


def plot_group_norms_vs_lambda(X, Y, grouped_indices_list, lambda_sorted, q_samples_sorted):
    dimension = len(grouped_indices_list)
    norm_flow = convert_q_to_group_norms(q_samples_sorted, grouped_indices_list)
    flow_norm_estimate = np.median(norm_flow, axis=1)

    # group = convert_group_indices_for_gglasso(grouped_indices_list)
    # W_glasso = glasso_path(X, Y, group, np.array(lambda_sorted))
    # lambda_glasso = np.array(W_glasso[4])
    # coeff_glasso = np.array(W_glasso[1])

    plot_title = 'Group Lasso Regression with CNF - Group Norms'

    colors = ['blue', 'green', 'red', 'brown', 'magenta', 'yellow', 'black', 'orange', 'purple', 'cyan']
    colors = list(islice(cycle(colors), dimension))
    my_linestyles = [
        'solid',
        (0, (1, 1)),
        'dashed',
        (0, (1, 10))]
    my_linestyles = list(islice(cycle(my_linestyles), dimension))

    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        y_value = flow_norm_estimate[:, index]
        plt.plot(lambda_sorted, y_value, color=colors[index], linestyle=my_linestyles[index], label=r'$\beta_{g' + str(index+1)+'}$')

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Group Norm')
    plt.title(plot_title)
    plt.legend()
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("./figures/Group_norms_vs_lambda" + plot_title.lower().replace(" ", "_") + ".pdf", dpi=300)
    return


def plot_flow_group_coefficients_path_vs_ground_truth(X, Y, lambda_sorted, tau_samples_sorted, solution_type):
    dimension = tau_samples_sorted[0][0].shape[0]
    x = lambda_sorted
    y = tau_samples_sorted

    p95 = np.quantile(y, 0.95, axis=1)
    p5 = np.quantile(y, 0.05, axis=1)
    means = np.mean(y, axis=1)

    colors = ['blue', 'green', 'red', 'brown', 'magenta', 'yellow', 'black', 'orange', 'purple', 'cyan']
    colors = list(islice(cycle(colors), dimension))

    plt.figure(figsize=(10, 5))

    for index in range(dimension):
        p95_np = p95[:, index]
        p5_np = p5[:, index]
        mean_np = means[:, index]
        plt.plot(x, mean_np, color=colors[index], label=r'$\tau^2_{' + str(index+1)+'}$')
        plt.fill_between(x, p5_np, p95_np, color=colors[index], alpha=0.1)

    plt.xlabel(r'$\lambda$')
    plt.xscale('log')
    plt.ylabel('Group Coefficients')
    plt.title('Group Lasso Regression with CNF - Group Coefficients Vs Lambda ' + solution_type)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/Group_Lasso_Group_coefficients_Vs_Lambda" + solution_type + ".pdf")
    plt.show()


def plot_log_marginal_likelihood_vs_lambda(X, Y, lambda_sorted, losses_sorted, variance, title="Distribution", grouped_indices_list=None, group_lasso=False):
    if grouped_indices_list is None:
        group = []
    else:
        group = convert_group_indices_for_gglasso(grouped_indices_list)

    n = Y.shape[0]
    y = -losses_sorted
    p95 = np.quantile(y, 0.95, axis=1)
    p5 = np.quantile(y, 0.05, axis=1)
    means = np.mean(y, axis=1)
    lambda_max_likelihood = lambda_sorted[np.argmax(means)]
    label_lambda_cv = ""

    if "ridge" in title.lower():
        alphas = lambda_sorted
        alpha_ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X, Y).alpha_
        print("Ridge CV lambda: ", alpha_ridge_cv)
        best_lambda_cv = 10 ** alpha_ridge_cv
        print("Ridge CV lambda rescaled: ", best_lambda_cv)
        label_lambda_cv = '$\lambda* RidgeCV$'
    elif "gl-without_betas" in title.lower():
        print()
        best_lambda_cv=""
    elif "group-lasso" in title.lower() or group_lasso:
        print()
        label_lambda_cv = '$\lambda* GLassoCV$'
        cv = cv_gglasso(X, Y.reshape(-1, 1), group, np.array(lambda_sorted))
        best_lambda_cv = cv[7]

    else:
        alphas = lambda_sorted * variance / n
        alpha_lasso_cv = LassoCV(alphas=alphas, cv=5).fit(X, Y).alpha_
        print("Lasso CV lambda: ", alpha_lasso_cv)
        best_lambda_cv = 10 ** (alpha_lasso_cv * n / variance)
        print("Lasso CV lambda rescaled: ", best_lambda_cv)
        label_lambda_cv = '$\lambda* LassoCV$'

    print("Best learnt lambda: ", lambda_max_likelihood)
    print("Best lambda from CV: ", best_lambda_cv)
    print("Max Log marginal likelihood: ", np.max(means))

    plt.figure(figsize=(10, 5))
    plt.plot(lambda_sorted, means)
    # plt.fill_between(lambda_sorted, p95, p5, color='g', alpha=0.3)

    plt.title('Log Marginal Likelihood vs Lambda - ' + title)
    # plt.xlabel(r'$\lambda$', fontsize=16)
    # plt.ylabel(r'$\log( p(X,Y \mid \lambda))$', fontsize=16)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Log Marginal Likelihood')
    plt.xscale('log')

    if label_lambda_cv != "":
        plt.vlines(best_lambda_cv, p5.min(), means.max(), label=label_lambda_cv, colors='r', linestyles='dashed')
    plt.vlines(lambda_max_likelihood, p5.min(), means.max(), label='$\lambda* Flow$', colors='b', linestyles='dashed')

    plt.legend()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("./figures/Log_Marginal_Likelihood-" + title + ".pdf")
    plt.show()

    f = open(f"./figures/Best_lambda_{title}.txt", "a")
    f.write(f"====================================================\n")
    f.write(f"Best_lambda_from_flows:{lambda_max_likelihood}\n")
    f.write(f"Best_lambda_{label_lambda_cv}:{best_lambda_cv}\n")
    f.write(f"====================================================\n")
    f.close()


def plot_t_distributions_for_each_invergamma_parameter_pairs(dimension, mean, scale_matrices, dfs, a_0, b_0, a_0np,
                                                             b_0np, q_samples, q_log_ps, flows, title=""):
    ranges = []
    sample_size = 50

    for i in range(dimension):
        ax_min = mean[i] - 3 * np.sqrt(scale_matrices[0][i, i])
        ax_max = mean[i] + 3 * np.sqrt(scale_matrices[0][i, i])
        points = np.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = np.meshgrid(*ranges, indexing='ij')
    rv = multivariate_t(loc=mean, shape=scale_matrices[0], df=dfs[0])
    analytical_density = rv.pdf(np.stack(grids, axis=-1))

    # grids = torch.from_numpy(grids)
    flow_inputs = torch.cat([torch.from_numpy(grid.reshape(-1, 1)) for grid in grids], dim=1)
    context = torch.cat((a_0, b_0), 1)
    flow_density = flows.log_prob(flow_inputs, context=context)
    reshape_argument = [sample_size for _ in range(dimension)]
    flow_density_reshaped = np.exp(flow_density[0].reshape(*reshape_argument))

    analytical_marginals = []
    flow_marginals = []
    for i in range(dimension):
        axes_to_sum = tuple(j for j in range(dimension) if j != i)
        analytical_marginal = analytical_density.sum(axis=axes_to_sum)
        flow_marginal = flow_density_reshaped.sum(axis=axes_to_sum)
        analytical_marginals.append(analytical_marginal)
        flow_marginals.append(flow_marginal)

    fig, axs = plt.subplots(1, dimension, figsize=(5 * dimension, 4))

    for i in range(dimension):
        axs[i].plot(ranges[i], analytical_marginals[i])
        axs[i].plot(ranges[i], flow_marginals[i])
        # axs[i].axvline(location_np[i], color='r', linestyle='--', label='Mean')
        axs[i].set_xlabel(r'$y$' + str(i))
        axs[i].set_ylabel("Value")
        axs[i].legend(['$Analytical$', '$Flow$'])

    plt.suptitle('Analytical Vs Flow P - T-distribution from Inv gamma for ' + str(a_0[0]) + " " + str(b_0[0]))
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_posterior_predictive_t_general_on_grid-" + str(a_0[0]) + "-" + str(
        b_0[0]) + ".pdf")
    plt.show()
    return

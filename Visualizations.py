import math

import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

sns.set_theme(style="darkgrid")


def plot_loss(loss_values):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(loss_values)), y=loss_values, color='blue', marker='o', markersize=5)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_analytical_flow_posterior_ridge_regression_on_grid(dimensions, mean_np, cov_np, flows):
    ranges = []
    sample_size = 200

    for i in range(dimensions):
        ax_min = mean_np[i] - 3 * torch.sqrt(cov_np[i, i])
        ax_max = mean_np[i] + 3 * torch.sqrt(cov_np[i, i])
        points = torch.linspace(ax_min, ax_max, sample_size)
        ranges.append(points)

    grids = torch.meshgrid(*ranges, indexing='ij')
    rv = multivariate_normal(mean_np, cov_np)
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

    plt.suptitle('Analytical Vs Flow Posterior - Ridge Regression')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_posterior_on_grid.pdf")
    plt.show()


def plot_analytical_flow_posterior_ridge_regression_with_samples(mean_np, cov_np,
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
    fig, axes = plt.subplots(1, num_axes, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if ax.axison:
            sns.histplot(data=data, x=i, ax=ax, hue="type", kde=True, bins=100)
            ax.legend(['analytical', 'flow'])
            ax.set_xlabel("W_" + str(i + 1))
            ax.set_ylabel("Value")

    plt.suptitle('Analytical Vs Flow Posterior - Ridge Regression')
    plt.tight_layout()
    plt.savefig("./figures/analytical_vs_flow_posterior_with_samples.pdf")
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
        sns.lineplot(x=0, y=i+1, data=data, label='W_'+str(i))

    plt.title('Beta Coefficients Vs Lambda - Lasso Regression')
    plt.xlabel(r'$\lambda$', fontsize=15)
    plt.xscale('log')

    plt.ylabel(r'$\beta$', fontsize=15)
    plt.tight_layout()
    plt.legend()
    plt.savefig("./figures/CNF_lasso_beta_vs_lambda.pdf")
    plt.show()

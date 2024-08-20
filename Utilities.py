import numpy as np
import torch


def extract_train_test_data(data_sample_size, train_ratio, X_full, Y_full):
    num_train = int(train_ratio * data_sample_size)
    num_test = data_sample_size - num_train
    indices = torch.randperm(data_sample_size)
    train_indices = indices[:num_train]
    test_indices = indices[num_test:]
    X_train, Y_train = X_full[train_indices], Y_full[train_indices]
    X_test, Y_test = X_full[test_indices], Y_full[test_indices]
    return X_train, Y_train, X_test, Y_test


def select_q_for_max_likelihood_lambda(lambda_max_likelihood, flows, device):
    lambda_max_likelihood_exp = np.log10(lambda_max_likelihood)
    uniform_lambdas = torch.zeros(1).to(device)
    lambdas_exp = (uniform_lambdas * 0 + lambda_max_likelihood_exp).view(-1, 1)
    context = lambdas_exp
    q_samples, q_log_prob = flows.sample_and_log_prob(num_samples=100, context=context)
    q_selected = q_samples.mean(dim=1)
    print("Coefficient selected from Flows : ", q_selected)
    return q_selected


def woodbury_matrix_conversion(A, U, C, V, device):
    # (A + UCV)^-1 = A^-1 - A^-1U(C^-1 + VA^-1U)^-1 VA^-1
    A_1 = torch.inverse(A)
    C_1 = torch.inverse(C)
    m = torch.matmul
    return A_1 - m(m(m(m(A_1, U), torch.inverse(C_1 + m(m(V, A_1), U))), V), A_1)
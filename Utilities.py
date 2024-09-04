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

    A_inv = torch.inverse(A)
    C_inv = torch.inverse(C)
    m = torch.matmul
    return A_inv - m(m(m(m(A_inv, U), torch.inverse(C_inv + m(m(V, A_inv), U))), V), A_inv)


def new_woodbury_identity(P, A, U, C, V, Q, device):
    # P(A + UCV)^-1Q    = PA^-1Q - PA^-1U(C^-1 + VA^-1U)^-1 VA^-1Q
    #                   = P(A^-1Q) - P(A^-1U)((C^-1 + V(A^-1U))^-1 V)(A^-1Q)
    # X = L^-1 M ==> LX = M
    A_inv_Q = torch.linalg.solve(A, Q).to(device)
    A_inv_U = torch.linalg.solve(A, U).to(device)
    C_inv = torch.inverse(C).to(device)

    m = torch.matmul
    term_1 = m(P, A_inv_Q)
    term_2 = m(P, A_inv_U)
    term_3 = torch.linalg.solve(C_inv + m(V, A_inv_U), V).to(device)
    term_4 = A_inv_Q
    result = term_1 - m(m(term_2, term_3), term_4)
    return result


def woodbury_identity_special(A, U, C, V, Q, device):
    # (A + UCV)^-1Q    = A^-1Q - A^-1U(C^-1 + VA^-1U)^-1 VA^-1Q
    #                   = (A^-1Q) - (A^-1U)((C^-1 + V(A^-1U))^-1 V)(A^-1Q)
    # X = L^-1 M ==> LX = M
    A_inv_Q = torch.linalg.solve(A, Q).to(device)
    A_inv_U = torch.linalg.solve(A, U).to(device)
    C_inv = torch.inverse(C).to(device)

    m = torch.matmul
    term_1 = torch.linalg.solve(C_inv + m(V, A_inv_U), V).to(device)
    result = A_inv_Q - m(m(A_inv_U, term_1), A_inv_Q)
    return result


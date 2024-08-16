import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import Visualizations as View


def evaluate_model(flows, q_selected, X_test, Y_test, type):
    Y_pred = torch.matmul(X_test, q_selected.squeeze(0))
    y_pred_np = Y_pred.detach().cpu().numpy()
    y_test_np = Y_test.detach().cpu().numpy()

    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    r2 = r2_score(y_test_np, y_pred_np)

    print("MSE : ", mse)
    print("RMSE : ", rmse)
    print("MAE : ", mae)
    print("R2 : ", r2)

    title = "Residuals - " + type
    View.plot_residuals(y_test_np, y_pred_np, title)


def evaluate_poisson_model(flows, q_selected, X_test, Z_test, type):
    Y_pred = torch.matmul(X_test, q_selected.squeeze(0))
    mean_poisson_pred = torch.exp(Y_pred)
    Z_pred = torch.poisson(mean_poisson_pred)

    z_pred_np = Z_pred.detach().cpu().numpy()
    z_test_np = Z_test.detach().cpu().numpy()

    mse = mean_squared_error(z_test_np, z_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(z_test_np, z_pred_np)
    r2 = r2_score(z_test_np, z_pred_np)

    print("MSE : ", mse)
    print("RMSE : ", rmse)
    print("MAE : ", mae)
    print("R2 : ", r2)

    title = "Residuals - " + type
    View.plot_residuals(z_test_np, z_pred_np, title)
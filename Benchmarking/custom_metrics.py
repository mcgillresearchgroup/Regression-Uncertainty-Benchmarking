import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer


def negative_log_likelihood(y_true, y_pred_mean, y_pred_var, eps=1e-6):
    """Calculate the negative log-likelihood for Gaussian distributed targets.
        Inputs:
            y_true: True target values
            y_pred_mean: Predicted mean values from the model
            y_pred_var: Predicted variance values from the model
            eps: Small constant to prevent division by zero in variance
        Returns:
            The negative log-likelihood.
        """
    y_pred_var = np.maximum(y_pred_var, eps)  # Ensure minimum variance
    nll = 0.5 * np.log(2*np.pi*y_pred_var) + ((y_true - y_pred_mean) ** 2) / (2 * y_pred_var)
    return np.mean(nll) 
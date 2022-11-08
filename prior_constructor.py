import numpy as np
from scipy.stats import multivariate_t


class prior_class:
    def __init__(self, data):
        num, _ = data.shape
        dim = 3
        pos_data = data[:, 0:2]
        nu_0 = dim + 2
        kappa_0 = 1
        sigma_0 = np.zeros((3, 3))
        sigma_0[0:2, 0:2] = np.cov(pos_data.T)
        sigma_0[2, 2] = 0.1
        lambda_0 = {
            "sigma_0": sigma_0 * (nu_0 - dim - 1),
            "nu_0": nu_0,
            "mu_0": np.hstack((np.mean(pos_data, axis=0), 0)),
            "kappa_0": kappa_0,
            }

        self.lambda_0 = lambda_0

    def posterior_predictive(self, data_tilde, data, direction_variance):
        lambda_0 = self.lambda_0
        num, dim = data.shape
        sample_mean = np.mean(data, axis=0)
        sample_var = 1 / num * (data - sample_mean).T @ (data - sample_mean)
        sample_var[-1, -1] = direction_variance

        sigma_0 = lambda_0["sigma_0"]
        nu_0 = lambda_0["nu_0"]
        mu_0 = lambda_0["mu_0"]
        kappa_0 = lambda_0["kappa_0"]
        lambda_N = {
            "sigma_N": sigma_0 + num * sample_var + kappa_0 * num / (kappa_0 + num) *
                       (sample_mean - mu_0)[:, np.newaxis] @ (sample_mean - mu_0)[np.newaxis, :],
            "nu_N": nu_0 + num,
            "mu_N": (kappa_0 * mu_0 + num * sample_mean) / (kappa_0 + num),
            "kappa_N": kappa_0 + num
        }
        student_t_shape = lambda_N["sigma_N"] * (lambda_N["kappa_N"] + 1) / (
                    lambda_N["kappa_N"] * (lambda_N["nu_N"] - dim + 1))
        likelihood = multivariate_t(loc=lambda_N["mu_N"], shape=student_t_shape,
                                    df=(lambda_N["nu_N"] - dim + 1), allow_singular=True).pdf(data_tilde)
        # likelihood = multivariate_normal(mean=lambda_N["mu_N"], cov=student_t_shape).pdf(data_tilde)
        return likelihood

    def prior_predictive(self, data_tilde):
        dim = data_tilde.shape[0]
        lambda_0 = self.lambda_0
        student_t_shape = lambda_0["sigma_0"] * (lambda_0["kappa_0"] + 1) / (
                lambda_0["kappa_0"] * (lambda_0["nu_0"] - dim + 1))
        likelihood = multivariate_t(loc=lambda_0["mu_0"], shape=student_t_shape,
                                    df=(lambda_0["nu_0"] - dim + 1)).pdf(data_tilde)
        # likelihood = multivariate_normal(mean=lambda_0["mu_0"], cov=student_t_shape).pdf(data_tilde)
        return likelihood

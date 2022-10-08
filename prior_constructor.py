import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from adaptive_kernel import trajectory_pre_processing

from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from scipy.special import multigammaln, gammaln


def kernel_density_estimator(data, if_plot=False):
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=2)
    grid.fit(data)
    kde_object = grid.best_estimator_
    if if_plot:
        nx, ny = (50, 50)
        X, Y = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        pdf = np.exp(kde_object.score_samples(xy))
        Z = pdf.reshape(ny, nx)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # plt.show()
    return kde_object


class mixture_prior:
    def __init__(self, data):
        assignment_array, parameter_list = trajectory_pre_processing(data)
        _, counts = np.unique(assignment_array, return_counts=True)
        weight_0_array = counts / np.sum(counts)
        lambda_0_list = []
        for index, value in enumerate(parameter_list):
            lambda_0_list.append({
                "sigma_0": value[1],
                "nu_0": 3,
                "mu_0": value[0],
                "kappa_0": 5,
                "weight_0": weight_0_array[index]
            })

        self.weight_0_array = weight_0_array
        self.lambda_0_list = lambda_0_list

    def rvs(self):
        """Draw a Random Variable Sample from the Mixture Prior Distribution"""
        cumulative_weight_array = np.cumsum(self.weight_0_array)
        draw = np.where(cumulative_weight_array > np.random.uniform(low=0, high=1))[0][0]
        sigma = invwishart(df=self.lambda_0_list[draw]["nu_0"], scale=self.lambda_0_list[draw]["sigma_0"]).rvs()
        mu = multivariate_normal(mean=self.lambda_0_list[draw]["mu_0"],
                                 cov=sigma / self.lambda_0_list[draw]["kappa_0"]).rvs()
        return mu, sigma

    def logpdf(self, mean, covariance):
        pdf = 0
        for _, value in enumerate(self.lambda_0_list):
            iw_log_pdf = invwishart(df=value["nu_0"], scale=value["sigma_0"]).logpdf(covariance)
            normal_log_pdf = multivariate_normal(mean=value["mu_0"], cov=covariance / value["kappa_0"]).logpdf(mean)
            pdf = pdf + np.exp(np.log(value["weight_0"]) + iw_log_pdf + normal_log_pdf)
        return np.log(pdf)

    def pdf(self, mean, covariance):
        pdf = 0
        for _, value in enumerate(self.lambda_0_list):
            iw_pdf = invwishart(df=value["nu_0"], scale=value["sigma_0"]).pdf(covariance)
            normal_pdf = multivariate_normal(mean=value["mu_0"], cov=covariance / value["kappa_0"]).pdf(mean)
            pdf = pdf + value["weight_0"] * iw_pdf * normal_pdf
        return pdf

    def posterior_rvs(self, data):
        """Closed form posterior of Conjugate Prior Mixture (No MCMC needed)"""
        lambda_0_list = self.lambda_0_list
        weight_0_array = self.weight_0_array
        lambda_N_list = []
        num = data.shape[0]
        sample_mean = np.mean(data, axis=0)
        sample_var = 1 / num * (data - sample_mean).T @ (data - sample_mean)
        for _, value in enumerate(lambda_0_list):
            sigma_0 = value["sigma_0"]
            nu_0 = value["nu_0"]
            mu_0 = value["mu_0"]
            kappa_0 = value["kappa_0"]
            lambda_N_list.append({
                "sigma_N": sigma_0 + num * sample_var + kappa_0 * num / (kappa_0 + num) *
                           (sample_mean - mu_0)[:, np.newaxis] @ (sample_mean - mu_0)[np.newaxis, :],
                "nu_N": nu_0 + num,
                "mu_N": (kappa_0 * mu_0 + num * sample_mean) / (kappa_0 + num),
                "kappa_N": kappa_0 + num
            })

        weight_N_array = np.zeros(len(lambda_0_list))
        for index, value in enumerate(lambda_N_list):
            weight_N_array[index] = marginal_log_likelihood(data, lambda_0_list[index], lambda_N_list[index])

        """
        Normalization of this new weight array
        """
        weight_N_array = min_max_normalization(weight_N_array) * weight_0_array
        weight_N_array = weight_N_array / np.sum(weight_N_array)
        # print(self.weight_0_array)
        # print(norm_weight_N_array)

        for index, value in enumerate(lambda_N_list):
            value["weight_N"] = weight_N_array[index]

        cumulative_weight_array = np.cumsum(weight_N_array)
        draw = np.where(cumulative_weight_array > np.random.uniform(low=0, high=1))[0][0]
        sigma = invwishart(df=lambda_N_list[draw]["nu_N"], scale=lambda_N_list[draw]["sigma_N"]).rvs()
        mu = multivariate_normal(mean=lambda_N_list[draw]["mu_N"],
                                 cov=sigma / lambda_N_list[draw]["kappa_N"]).rvs()

        return mu, sigma


def min_max_normalization(array):
    max_value = np.max(array)
    min_value = np.min(array)
    norm_array = np.zeros(array.shape)
    for index, value in enumerate(array):
        norm_array[index] = (value - min_value) / (max_value - min_value)
    norm_array = norm_array / np.sum(norm_array)
    return norm_array


def marginal_log_likelihood(data, lambda_0, lambda_N):
    """Credit to Haihui's Python Implementation of phy-gmm"""
    N = data.shape[0]
    M = 2
    logLik = multigammaln(0.5 * lambda_N["nu_N"], M) - multigammaln(0.5 * lambda_0["nu_0"], M) + lambda_0[
        "nu_0"] / 2 * logDet(lambda_0["sigma_0"]) - lambda_N["nu_N"] / 2 * logDet(
        lambda_N["sigma_N"]) + 0.5 * M * np.log(lambda_0["kappa_0"] / lambda_N["kappa_N"]) - 0.5 * N * M * np.log(np.pi)
    return logLik


def logDet(matrix):
    """Credit to Haihui's Python Implementation of phy-gmm"""
    sign, logdet = np.linalg.slogdet(matrix)
    return sign * logdet


if __name__ == "__main__":
    """2D KDE"""
    import scipy.io as scio

    # np.random.seed(1)
    # N = 200
    # X1 = np.random.multivariate_normal([-2, -2], [[2, 1], [1, 1]], int(0.2 * N))
    # X2 = np.random.multivariate_normal([2, 2], [[2, 0], [0, 0.1]], int(0.8 * N))
    # X = np.concatenate((X1, X2))
    Data = scio.loadmat('Increm_Learning/linearity.mat')['Xi_ref'].T
    Data = Data
    prior = mixture_prior(Data)
    a, b = prior.posterior_rvs(Data[130:150, :])

    # lambda_0 = {"sigma_0": np.cov(Data.T),
    #             "nu_0": 3,
    #             "mu_0": np.mean(Data, axis=0),
    #             "kappa_0": 5}
    #
    # sigma_0 = lambda_0["sigma_0"]
    # nu_0 = lambda_0["nu_0"]
    # mu_0 = lambda_0["mu_0"]
    # kappa_0 = lambda_0["kappa_0"]
    # num = Data.shape[0]
    # sample_mean = np.mean(Data, axis=0)
    # sample_var = 1 / num * (Data - sample_mean).T @ (Data - sample_mean)
    #
    # lambda_N = {
    #     "sigma_N": sigma_0 + num * sample_var + kappa_0 * num / (kappa_0 + num) *
    #                (sample_mean - mu_0)[:, np.newaxis] @ (sample_mean - mu_0)[np.newaxis, :],
    #     "nu_N": nu_0 + num,
    #     "mu_N": (kappa_0 * mu_0 + num * sample_mean) / (kappa_0 + num),
    #     "kappa_N": kappa_0 + num
    # }
    #
    # logpdf = marginal_likelihood(Data, lambda_0, lambda_N)
    # print(logpdf)

    # print(X.shape)
    # kde = kernel_density_estimator(X)
    # # Plot PDF of estimated distribution
    # nx, ny = (50, 50)
    # X, Y = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))
    # xy = np.vstack([X.ravel(), Y.ravel()]).T
    # pdf = np.exp(kde.score_samples(xy))
    # Z = pdf.reshape(ny, nx)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    # plt.show()

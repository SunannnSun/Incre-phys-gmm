import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time

from adaptive_kernel import trajectory_pre_processing
from spectral_clustering import spectral_clustering

from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from scipy.special import multigammaln, gammaln


class kde_prior:
    def __init__(self, data, *vel_data, option="full", if_plot=False):
        if option == "full":
            params = {"bandwidth": np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2)
            grid.fit(data)
            kde_object = grid.best_estimator_
        elif option == "spectral":
            assignment_array, parameter_list = spectral_clustering(data, vel_data, if_plot=True)
            theta_array = np.zeros((assignment_array.shape[0], 2))
            for index in range(theta_array.shape[0]):
                theta_array[index, :] = parameter_list[assignment_array[index]][0]
            params = {"bandwidth": np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2)
            grid.fit(theta_array)
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
        self.kde_object_list = [kde_object]

    def update(self, assignment_array, parameter_list):
        data = np.zeros((assignment_array.shape[0], 2))
        for index in range(data.shape[0]):
            data[index, :] = parameter_list[assignment_array[index]][0]
        params = {"bandwidth": np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=2)
        grid.fit(data)
        kde_object = grid.best_estimator_
        self.kde_object_list.append(kde_object)

    def rvs(self):
        rand = np.random.randint(0, len(self.kde_object_list))
        new_mu = self.kde_object_list[rand].sample()[0]
        return new_mu

    def posterior_rvs(self, data, data_cov):
        dim = data_cov.shape[0]
        iteration = 100
        x_old = np.random.multivariate_normal(np.ones((dim,)) , np.eye(dim))
        # print(x_old)
        pdf = 0
        for kde_object in self.kde_object_list:
            pdf = pdf + 1 / len(self.kde_object_list) * np.exp(
                np.clip(kde_object.score_samples(x_old[np.newaxis, :]), -50, 50))

        f_old = np.sum(multivariate_normal.logpdf(data, mean=x_old, cov=data_cov)) + np.log(pdf)
        # x_hist = x_old[np.newaxis, :]
        for t in range(iteration):
            random_walk_cov = 0.25 * np.eye(dim)
            x_new = multivariate_normal.rvs(mean=x_old, cov=random_walk_cov)
            pdf = 0
            for kde_object in self.kde_object_list:
                pdf = pdf + 1 / len(self.kde_object_list) * np.exp(
                    np.clip(kde_object.score_samples(x_new[np.newaxis, :]), -50, 50))
            f_new = np.sum(multivariate_normal.logpdf(data, mean=x_new, cov=data_cov)) + np.log(pdf)
            a_ratio = min([1, np.exp(np.clip(f_new - f_old, -50, 50))])
            if np.random.uniform(0, 1) <= a_ratio:
                x_old = x_new
                f_old = f_new
            # x_hist = np.append(x_hist, x_old[np.newaxis, :], axis=0)
        return x_old


class mixture_prior:
    def __init__(self, data):
        num, dim = data.shape
        assignment_array, parameter_list = trajectory_pre_processing(data, one_prior=True)
        # assignment_array, parameter_list = spectral_clustering(pos_data, vel_data)
        _, counts = np.unique(assignment_array, return_counts=True)
        # weight_0_array = counts / np.sum(counts)
        weight_0_array = 1 / counts.shape[0] * np.ones(counts.shape[0])
        lambda_0_list = []

        for index, value in enumerate(parameter_list):
            nu_0 = dim + 2
            kappa_0 = 1
            lambda_0_list.append({
                "sigma_0": value[1] * (nu_0 - dim - 1),
                "nu_0": nu_0,
                "mu_0": value[0],
                "kappa_0": kappa_0,
                "weight_0": weight_0_array[index]
            })

        self.weight_0_array = weight_0_array
        self.lambda_0_list = lambda_0_list

    # def rvs(self):
    #     """Draw a Random Variable Sample from the Mixture Prior Distribution"""
    #     cumulative_weight_array = np.cumsum(self.weight_0_array)
    #     draw = np.where(cumulative_weight_array > np.random.uniform(low=0, high=1))[0][0]
    #     # sigma = invwishart(df=self.lambda_0_list[draw]["nu_0"], scale=self.lambda_0_list[draw]["sigma_0"]).rvs()
    #     # mu = multivariate_normal(mean=self.lambda_0_list[draw]["mu_0"],
    #     #                          cov=sigma / self.lambda_0_list[draw]["kappa_0"]).rvs()
    #     sigma = invwishart(df=3, scale=self.lambda_0_list[draw]["sigma_0"]).rvs()
    #     mu = multivariate_normal(mean=self.lambda_0_list[draw]["mu_0"],
    #                              cov=sigma).rvs()
    #     return mu, sigma
    #
    # def logpdf(self, mean, covariance):
    #     pdf = 0
    #     for _, value in enumerate(self.lambda_0_list):
    #         iw_log_pdf = invwishart(df=value["nu_0"], scale=value["sigma_0"]).logpdf(covariance)
    #         normal_log_pdf = multivariate_normal(mean=value["mu_0"], cov=covariance / value["kappa_0"]).logpdf(mean)
    #         pdf = pdf + np.exp(np.log(value["weight_0"]) + iw_log_pdf + normal_log_pdf)
    #     return np.log(pdf)
    #
    # def pdf(self, mean, covariance):
    #     pdf = 0
    #     for _, value in enumerate(self.lambda_0_list):
    #         iw_pdf = invwishart(df=value["nu_0"], scale=value["sigma_0"]).pdf(covariance)
    #         normal_pdf = multivariate_normal(mean=value["mu_0"], cov=covariance / value["kappa_0"]).pdf(mean)
    #         pdf = pdf + value["weight_0"] * iw_pdf * normal_pdf
    #     return pdf
    #
    # def posterior_rvs(self, data):
    #     """Closed form posterior of Conjugate Prior Mixture (No MCMC needed)"""
    #     lambda_0_list = self.lambda_0_list
    #     weight_0_array = self.weight_0_array
    #     lambda_N_list = []
    #     num = data.shape[0]
    #     sample_mean = np.mean(data, axis=0)
    #     sample_var = 1 / num * (data - sample_mean).T @ (data - sample_mean)
    #     for _, value in enumerate(lambda_0_list):
    #         sigma_0 = value["sigma_0"]
    #         nu_0 = value["nu_0"]
    #         mu_0 = value["mu_0"]
    #         kappa_0 = value["kappa_0"]
    #         lambda_N_list.append({
    #             "sigma_N": sigma_0 + num * sample_var + kappa_0 * num / (kappa_0 + num) *
    #                        (sample_mean - mu_0)[:, np.newaxis] @ (sample_mean - mu_0)[np.newaxis, :],
    #             "nu_N": nu_0 + num,
    #             "mu_N": (kappa_0 * mu_0 + num * sample_mean) / (kappa_0 + num),
    #             "kappa_N": kappa_0 + num
    #         })
    #
    #     weight_N_array = np.zeros(len(lambda_0_list))
    #     for index, value in enumerate(lambda_N_list):
    #         weight_N_array[index] = marginal_log_likelihood(data, lambda_0_list[index], lambda_N_list[index])
    #
    #     """
    #     Normalization of this new weight array
    #     """
    #     weight_N_array = min_max_normalization(weight_N_array) * weight_0_array
    #
    #     weight_N_array = weight_N_array / np.sum(weight_N_array)
    #     # print(self.weight_0_array)
    #     # print(weight_N_array)
    #
    #     for index, value in enumerate(lambda_N_list):
    #         value["weight_N"] = weight_N_array[index]
    #
    #     cumulative_weight_array = np.cumsum(weight_N_array)
    #     if len(weight_N_array) == 1:
    #         draw = 0
    #     else:
    #         draw = np.where(cumulative_weight_array > np.random.uniform(low=0, high=1))[0][0]
    #     sigma = invwishart(df=lambda_N_list[draw]["nu_N"], scale=lambda_N_list[draw]["sigma_N"]).rvs()
    #     mu = multivariate_normal(mean=lambda_N_list[draw]["mu_N"],
    #                              cov=sigma / lambda_N_list[draw]["kappa_N"]).rvs()
    #
    #     return mu, sigma
    #
    # def posterior_predictive(self, data_tilde, data):
    #
    #     # start = time.time()
    #     lambda_0_list = self.lambda_0_list
    #     weight_0_array = self.weight_0_array
    #     lambda_N_list = []
    #     num = data.shape[0]
    #     sample_mean = np.mean(data, axis=0)
    #     sample_var = 1 / num * (data - sample_mean).T @ (data - sample_mean)
    #     for _, value in enumerate(lambda_0_list):
    #         sigma_0 = value["sigma_0"]
    #         nu_0 = value["nu_0"]
    #         mu_0 = value["mu_0"]
    #         kappa_0 = value["kappa_0"]
    #         lambda_N_list.append({
    #             "sigma_N": sigma_0 + num * sample_var + kappa_0 * num / (kappa_0 + num) *
    #                        (sample_mean - mu_0)[:, np.newaxis] @ (sample_mean - mu_0)[np.newaxis, :],
    #             "nu_N": nu_0 + num,
    #             "mu_N": (kappa_0 * mu_0 + num * sample_mean) / (kappa_0 + num),
    #             "kappa_N": kappa_0 + num
    #         })
    #
    #     weight_N_array = np.zeros(len(lambda_0_list))
    #     likelihood = np.zeros(len(lambda_0_list))
    #
    #     for index, value in enumerate(lambda_N_list):
    #         weight_N_array[index] = marginal_log_likelihood(data, lambda_0_list[index], lambda_N_list[index])
    #         student_t_shape = value["sigma_N"] * (value["kappa_N"] + 1) / (value["kappa_N"] * (value["nu_N"] - 2 + 1))
    #         likelihood[index] = multivariate_t(loc=value["mu_N"], shape=student_t_shape,
    #                                            df=(value["nu_N"] - 2 + 1)).pdf(data_tilde)
    #     # print("Likelihood array:", likelihood)
    #     """
    #     Normalization of this new weight array
    #     """
    #     weight_N_array = weight_N_array / np.sum(weight_N_array)
    #     # print("weight array:", weight_N_array)
    #     likelihood = likelihood * weight_N_array
    #     likelihood = np.sum(likelihood)
    #     # end = time.time()
    #     # print("Initialization Done. Time taken: %.2f sec" % (end - start))
    #     return likelihood

    def posterior_predictive(self, data_tilde, data):
        lambda_0 = self.lambda_0_list[0]

        num, dim = data.shape

        sample_mean = np.mean(data, axis=0)
        sample_var = 1 / num * (data - sample_mean).T @ (data - sample_mean)
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
        # print(student_t_shape)
        likelihood = multivariate_t(loc=lambda_N["mu_N"], shape=student_t_shape,
                                    df=(lambda_N["nu_N"] - dim + 1), allow_singular=True).pdf(data_tilde)
        # likelihood = multivariate_normal(mean=lambda_N["mu_N"], cov=student_t_shape).pdf(data_tilde)
        return likelihood

    def prior_predictive(self, data_tilde):
        lambda_0 = self.lambda_0_list[0]
        student_t_shape = lambda_0["sigma_0"] * (lambda_0["kappa_0"] + 1) / (lambda_0["kappa_0"] * (lambda_0["nu_0"] - 2 + 1))
        likelihood = multivariate_t(loc=lambda_0["mu_0"], shape=student_t_shape,
                                    df=(lambda_0["nu_0"] - 2 + 1)).pdf(data_tilde)
        return likelihood


def min_max_normalization(array):
    if array.shape[0] == 1:
        return array
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
    # return logLik
    return np.exp(logLik)


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
    Data = Data[np.arange(0, Data.shape[0], 2)]
    Data = Data[220:-1, :]
    prior = kde_prior(Data)
    C_array, phi_list = trajectory_pre_processing(Data)
    print("phi list: ", phi_list)
    prior.update(C_array, phi_list)

    # print(prior.rvs())
    # print(prior.posterior_rvs(Data, np.cov(Data.T)))
    # print(prior.lambda_0_list)
    trajectory_pre_processing(Data)

    # prior.prior_predictive(Data[25, :])

    plt.show()

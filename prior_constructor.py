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
        num, _ = data.shape
        dim = 3
        assignment_array, parameter_list = trajectory_pre_processing(data[:, 0:2], one_prior=True)
        # assignment_array, parameter_list = spectral_clustering(pos_data, vel_data)
        _, counts = np.unique(assignment_array, return_counts=True)
        # weight_0_array = counts / np.sum(counts)
        weight_0_array = 1 / counts.shape[0] * np.ones(counts.shape[0])
        lambda_0_list = []
        if data.shape[1] == 2:
            for index, value in enumerate(parameter_list):
                nu_0 = dim + 2
                kappa_0 = 1
                # sigma_0 = np.diag([0.01, 0.01, np.pi])
                lambda_0_list.append({
                    "sigma_0": value[1] * (nu_0 - dim - 1),
                    "nu_0": nu_0,
                    "mu_0": value[0],
                    "kappa_0": kappa_0,
                    "weight_0": weight_0_array[index]
                })
        else:
            for index, value in enumerate(parameter_list):
                nu_0 = dim + 2
                kappa_0 = 1
                sigma_0 = np.zeros((3, 3))
                sigma_0[0:2, 0:2] = value[1] / 5
                # print(value[1])
                sigma_0[2, 2] = 0.5
                lambda_0_list.append({
                    "sigma_0": sigma_0 * (nu_0 - dim - 1),
                    "nu_0": nu_0,
                    "mu_0": np.hstack((value[0], 0)),
                    "kappa_0": kappa_0,
                    "weight_0": weight_0_array[index]
                })
        self.weight_0_array = weight_0_array
        self.lambda_0_list = lambda_0_list

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
            # "sigma_N": sigma_0 + num * sample_var,
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
        lambda_0 = self.lambda_0_list[0]
        student_t_shape = lambda_0["sigma_0"] * (lambda_0["kappa_0"] + 1) / (
                lambda_0["kappa_0"] * (lambda_0["nu_0"] - dim + 1))
        likelihood = multivariate_t(loc=lambda_0["mu_0"], shape=student_t_shape,
                                    df=(lambda_0["nu_0"] - dim + 1)).pdf(data_tilde)

        # likelihood = multivariate_normal(mean=lambda_0["mu_0"], cov=student_t_shape).pdf(data_tilde)
        return likelihood




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

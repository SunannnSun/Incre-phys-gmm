import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from KDE import kernel_density_estimator
import scipy.io as scio


def gibbs_sampling():
    return None


def draw_new_theta(kde_object, hyperparameter):
    new_mu    = kde_object.sample()[0]
    new_Sigma = invwishart(df=hyperparameter["nu_0"], scale=hyperparameter["Sigma_0"]).rvs()
    return new_mu, new_Sigma


def draw_posterior_inverse_wishart(data, hyperparameter):
    num = data.shape[0]
    x_bar = np.mean(data, axis=0)
    sample_var = 1 / num * (data - x_bar).T @ (data - x_bar)
    Sigma_N = hyperparameter["Sigma_0"] + num * sample_var
    nu_N = hyperparameter["nu_0"] + num
    iw_N = invwishart(df=nu_N, scale=Sigma_N)
    return iw_N.rvs()


def draw_posterior_kde(data, data_cov, kde_object):
    iteration = 200
    x_old = np.array([0, 0])
    f_old = np.sum(multivariate_normal.logpdf(data, mean=x_old, cov=data_cov)) + kde_object.score_samples(
        x_old[np.newaxis, :])
    for t in range(iteration):
        x_new = multivariate_normal.rvs(mean=x_old, cov=0.25*np.identity(2))
        f_new = np.sum(multivariate_normal.logpdf(data, mean=x_new, cov=data_cov)) + kde_object.score_samples(
            x_new[np.newaxis, :])
        a_ratio = min([1, np.exp(np.clip(f_new-f_old, -50, 50))])
        if np.random.uniform(0, 1) <= a_ratio:
            x_old = x_new
            f_old = f_new
    return x_old


if __name__ == "__main__":

    # Data = np.random.multivariate_normal(mean=[-2, 2], cov=[[1, 0], [0, 1]], size=10)
    # np.random.seed(1)
    N = 10
    X1 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], int(N))
    X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], int(0.7 * N))
    Data = np.concatenate((X1, X2))
    # Data = X1
    # Data = scio.loadmat('Increm_Learning/small_stair.mat')['Xi_ref'].T

    # fig, ax = plt.subplots()
    # ax.scatter(Data[:, 0], Data[:, 1])
    # plt.show()
    (N, M) = Data.shape

    """ Hyperparameter and Base Distribution """
    lambda_0 = {
        "nu_0": 3,
        "Sigma_0": np.array([[1, 0], [0, 1]])
    }
    kde = kernel_density_estimator(Data)

    """ Initialize the Gibbs Sampling for First Iteration"""
    num_aux = 2
    alpha = 2
    phi_list = []
    for i in range(N):
        sigma = draw_posterior_inverse_wishart(Data[i, :][np.newaxis, :], lambda_0)
        mu = draw_posterior_kde(Data[i, :], sigma, kde)
        phi_list.append((mu, sigma))
    C_array = np.arange(N)

    """ Sampling """
    # Current assignment array
    for sampling_iter in range(40):
        for n in np.random.permutation(N):
            # Delete the current c_i and count distinct c_j
            values, counts = np.unique(np.delete(C_array, n), return_counts=True)
            # Shape of values or counts array is exactly the number of distinct c_j, denoted as K
            K = values.shape[0]
            cond_prob = np.zeros(K+num_aux)
            # Populate the conditional probability for each possible phi_c_k
            for k in range(K):
                c_k = values[k]
                # Unpack phi_c_k, phi_list should always have the length of either K or K+1
                mu_c_k, sigma_c_k = phi_list[c_k]
                # calculate the conditional probability for c_i = c_k
                cond_prob[k] = counts[k]/(N + alpha - 1) * multivariate_normal.pdf(x=Data[n, :], mean=mu_c_k, cov=sigma_c_k)
            phi_aux_list = []
            for k in range(num_aux):
                # Draw new theta for auxiliary variables
                mu_c_k, sigma_c_k = draw_new_theta(kde, lambda_0)
                phi_aux_list.append((mu_c_k, sigma_c_k))
                cond_prob[K+k] = (alpha/num_aux)/(N + alpha - 1) * multivariate_normal.pdf(x=Data[n, :], mean=mu_c_k, cov=sigma_c_k)

            # Sample a new c_i
            cond_prob = cond_prob / np.sum(cond_prob)
            cond_prob = np.cumsum(cond_prob)
            draw = np.where(cond_prob > np.random.uniform(low=0, high=1))[0][0]
            if draw < K:
                C_array[n] = values[draw]
            else:
                C_array[n] = K - 1 - draw  # this will give -1 or -2 for auxiliary variables
            # Adjust assignment array and Phi
            rearrange_list = []
            for i in range(N):
                if C_array[i] not in rearrange_list:
                    rearrange_list.append(C_array[i])
                    C_array[i] = rearrange_list.index(C_array[i])
                else:
                    C_array[i] = rearrange_list.index(C_array[i])
            # print(rearrange_list)
            phi_list_new = []
            for i, element in enumerate(rearrange_list):
                if element >= 0:
                    phi_list_new.append(phi_list[rearrange_list[i]])
                else:
                    phi_list_new.append(phi_aux_list[abs(element)-1])
            phi_list = phi_list_new

        # Draw new values for each phi given prior and data associated with each phi
        for k in range(len(phi_list)):
            x_k = Data[C_array == k]
            sigma_k = draw_posterior_inverse_wishart(x_k, lambda_0)
            mu_k = draw_posterior_kde(x_k, sigma_k, kde)
            (mu_k, sigma_k) = phi_list[k]

        # print(C_array)

    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", 'c', 'm']
    for i in range(N):
        color = colors[C_array[i]]
        ax.scatter(Data[i, 0], Data[i, 1], c=color)
    plt.show()














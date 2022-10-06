import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
import scipy.io as scio
from KDE import kernel_density_estimator


def gibbs_sampling(data, assignment_array, parameter_list, mu_prior, cov_prior, num_aux=5, alpha=1):

    (N, M) = data.shape

    for i in np.random.permutation(N):
        # Delete the current c_i and count distinct c_j for j != i
        values, counts = np.unique(np.delete(assignment_array, i), return_counts=True)
        # Length of values or counts vector is exactly the number of distinct c, denoted as K
        K = values.shape[0]
        cond_prob = np.zeros(K + num_aux)
        # Populate the conditional probability for each possible phi_c_k
        for k, c_k in enumerate(values):
            # Unpack phi_c_k
            mu_c_k, sigma_c_k = parameter_list[c_k]
            # calculate the conditional probability for c_i = c_k
            cond_prob[k] = counts[k] / (N + alpha - 1) * multivariate_normal.pdf(x=data[i, :], mean=mu_c_k,
                                                                                 cov=sigma_c_k)
        aux_parameter_list = []
        for k in range(num_aux):
            # Draw new theta for auxiliary variables
            mu_c_k, sigma_c_k = draw_new_theta(mu_prior, cov_prior)
            aux_parameter_list.append((mu_c_k, sigma_c_k))
            cond_prob[K + k] = (alpha / num_aux) / (N + alpha - 1) * multivariate_normal.pdf(x=data[i, :],
                                                                                             mean=mu_c_k,
                                                                                             cov=sigma_c_k)

        # Draw a new c_i
        cond_prob = cond_prob / np.sum(cond_prob)
        cond_prob = np.cumsum(cond_prob)
        draw = np.where(cond_prob > np.random.uniform(low=0, high=1))[0][0]
        if draw < K:
            assignment_array[i] = values[draw]
        else:
            assignment_array[i] = K - 1 - draw  # this gives negative value for auxiliary variables

        # Re-order the assignment_array and parameter_list
        rearrange_list = []
        for index, value in enumerate(assignment_array):
            if value not in rearrange_list:
                rearrange_list.append(value)
                assignment_array[index] = rearrange_list.index(value)
            else:
                assignment_array[index] = rearrange_list.index(value)
        new_parameter_list = []
        for index, value in enumerate(rearrange_list):
            if value >= 0:
                new_parameter_list.append(parameter_list[rearrange_list[index]])
            else:
                new_parameter_list.append(aux_parameter_list[abs(value) - 1])  # update param from aux list
        parameter_list = new_parameter_list

    # Draw new phi given prior and associated data
    for k in range(len(parameter_list)):
        x_k = data[assignment_array == k]
        sigma_k = draw_posterior_inverse_wishart(x_k, cov_prior)
        mu_k = draw_posterior_kde(x_k, sigma_k, mu_prior)
        parameter_list[k] = (mu_k, sigma_k)

    return assignment_array, parameter_list


def draw_new_theta(kde_object, hyperparameter):
    new_mu = kde_object.sample()[0]
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
        x_new = multivariate_normal.rvs(mean=x_old, cov=0.2 * np.identity(2))
        f_new = np.sum(multivariate_normal.logpdf(data, mean=x_new, cov=data_cov)) + kde_object.score_samples(
            x_new[np.newaxis, :])
        a_ratio = min([1, np.exp(np.clip(f_new - f_old, -50, 50))])
        if np.random.uniform(0, 1) <= a_ratio:
            x_old = x_new
            f_old = f_new
    return x_old

if __name__ == "__main__":

    """Initialize Data"""
    # np.random.seed(1)
    # num_data = 20
    # X1 = np.random.multivariate_normal([0, 0], [[15, 0.1], [0.1, 0.1]], int(0.3* num_data))
    # X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], int(0.7 * num_data))
    # Data = np.concatenate((X1, X2))
    # Data = X2
    Data = scio.loadmat('Increm_Learning/S_shape.mat')['Xi_ref'].T

    """Initialize Prior"""
    lambda_0 = {
        "nu_0": 3,
        "Sigma_0": np.array([[1, 0], [0, 1]])
    }
    kde = kernel_density_estimator(Data, True)

    """Initial Guess of assignment_array and parameter_list"""
    phi_list = []
    for i in range(Data.shape[0]):
        sigma = draw_posterior_inverse_wishart(Data[i, :][np.newaxis, :], lambda_0)
        mu = draw_posterior_kde(Data[i, :], sigma, kde)
        phi_list.append((mu, sigma))
    C_array = np.arange(Data.shape[0])

    """Begin Gibbs Sampling"""
    for iteration in range(50):
        C_array, phi_list = gibbs_sampling(Data, C_array, phi_list, kde, lambda_0)

    """Plot results"""
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", 'c', 'm', 'w', 'y', 'crimson', 'lime']
    for i in range(Data.shape[0]):
        color = colors[C_array[i]]
        ax.scatter(Data[i, 0], Data[i, 1], c=color)
    plt.show()

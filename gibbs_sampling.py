from prior_constructor import *


def gibbs_sampling(data, assignment_array, parameter_list, *prior_distribution, num_aux=2, alpha=0.001):
    (N, M) = data.shape
    if len(prior_distribution) == 2:
        mu_prior = prior_distribution[0]
        cov_prior = prior_distribution[1]

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
            cond_prob[k] = counts[k] / (N + alpha - 1) * multivariate_normal.logpdf(x=data[i, :], mean=mu_c_k,
                                                                                 cov=sigma_c_k)
        aux_parameter_list = []
        for k in range(num_aux):
            # Draw new theta for auxiliary variables
            mu_c_k, sigma_c_k = draw_new_theta(prior_distribution)
            aux_parameter_list.append((mu_c_k, sigma_c_k))
            # print(data[i, :])
            cond_prob[K + k] = (alpha / num_aux) / (N + alpha - 1) * multivariate_normal.logpdf(x=data[i, :],
                                                                                             mean=mu_c_k,
                                                                                             cov=sigma_c_k)

        # Draw a new c_i
        cond_prob = cond_prob / np.sum(cond_prob)
        cond_prob = np.cumsum(cond_prob)
        rand = np.random.uniform(low=0, high=1)

        draw = np.where(cond_prob > rand)[0][0]
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
        if len(prior_distribution) == 2:
            sigma_k = draw_posterior_inverse_wishart(x_k, cov_prior)
            # mu_k = draw_posterior_kde(x_k, sigma_k, mu_prior)
            # print(sigma_k)
            mu_k = mu_prior.posterior_rvs(x_k, sigma_k)

        else:
            mu_k, sigma_k = prior_distribution[0].posterior_rvs(x_k)
        parameter_list[k] = (mu_k, sigma_k)

    return assignment_array, parameter_list


def draw_new_theta(prior_distribution):
    if len(prior_distribution) == 2:
        kde_object = prior_distribution[0]
        hyperparameter = prior_distribution[1]
        # new_mu = kde_object.sample()[0]
        new_mu = kde_object.rvs()
        new_Sigma = invwishart(df=hyperparameter["nu_0"], scale=hyperparameter["Sigma_0"]).rvs()
    else:
        prior_distribution = prior_distribution[0]
        new_mu, new_Sigma = prior_distribution.rvs()
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
    """Adaptive Metropolis Algorithm to approximate the Posterior of KDE"""
    beta = 0.05
    s_d = 2.38 ** 2 / 2
    iteration = 100
    x_old = np.random.multivariate_normal([0, 0], np.identity(2))
    f_old = np.sum(multivariate_normal.logpdf(data, mean=x_old, cov=data_cov)) + kde_object.score_samples(
        x_old[np.newaxis, :])
    x_hist = x_old[np.newaxis, :]
    for t in range(iteration):
        if True:
            random_walk_cov = 0.25 * np.identity(2)
        else:
            # k = x_hist.shape[0]
            # empirical_cov = 1 / (k-1)  * (x_hist.T @ x_hist - k * np.mean(x_hist, axis=0)[np.newaxis, :].T
            # @ np.mean(x_hist, axis=0)[np.newaxis, :])
            random_walk_cov = (1 - beta) * s_d * np.cov(x_hist.T) + beta * (0.1 ** 2 / 2 * np.identity(2))
        x_new = multivariate_normal.rvs(mean=x_old, cov=random_walk_cov)
        f_new = np.sum(multivariate_normal.logpdf(data, mean=x_new, cov=data_cov)) + kde_object.score_samples(
            x_new[np.newaxis, :])
        a_ratio = min([1, np.exp(np.clip(f_new - f_old, -50, 50))])
        if np.random.uniform(0, 1) <= a_ratio:
            x_old = x_new
            f_old = f_new
        x_hist = np.append(x_hist, x_old[np.newaxis, :], axis=0)
    return x_old


if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], [[15, 0.1], [0.1, 0.1]], int(0.3 * 50))
    X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], int(0.7 * 50))
    Prior = np.concatenate((X1, X2))

    Data = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], int(200))


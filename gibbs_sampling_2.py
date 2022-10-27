import time

from prior_constructor import *


def gibbs_sampling_2(data, assignment_array, prior_distribution, alpha):
    pos_data = data[:, 0:2]
    vel_data = data[:, 2:4]

    (N, M) = pos_data.shape
    for i in np.random.permutation(N):
        pos_data_exclude_i = np.delete(pos_data, i, axis=0)
        vel_data_exclude_i = np.delete(vel_data, i, axis=0)
        data_exclude_i = np.delete(data, i, axis=0)

        assignment_array_exclude_i = np.delete(assignment_array, i)

        values, counts = np.unique(assignment_array_exclude_i, return_counts=True)

        K = values.shape[0]
        cond_prob = np.zeros(K+1)
        # Populate the conditional probability for each k in K

        for k, c_k in enumerate(values):
            # calculate the conditional probability for c_i = c_k
            data_post_pred = prior_distribution.posterior_predictive(data_tilde=data[i, :],
                                                                     data=data_exclude_i[
                                                                        assignment_array_exclude_i == c_k])
            cond_prob[k] = counts[k] / (N - 1 + alpha) * data_post_pred

        cond_prob[-1] = alpha / (N - 1 + alpha) * prior_distribution.prior_predictive(data_tilde=data[i, :])

        # Draw a new c_i
        cond_prob = cond_prob / np.sum(cond_prob)
        cond_prob = np.cumsum(cond_prob)
        rand = np.random.uniform(low=0, high=1)
        draw = np.where(cond_prob > rand)[0][0]
        if draw < K:
            assignment_array[i] = values[draw]
        else:
            assignment_array[i] = K
        # Re-order the assignment_array and parameter_list
        rearrange_list = []
        for index, value in enumerate(assignment_array):
            if value not in rearrange_list:
                rearrange_list.append(value)
                assignment_array[index] = rearrange_list.index(value)
            else:
                assignment_array[index] = rearrange_list.index(value)

    return assignment_array


if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], [[15, 0.1], [0.1, 0.1]], int(3))
    print(X1)
    print(np.delete(X1, 0, axis=0))

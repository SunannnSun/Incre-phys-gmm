import numpy as np


def karcher_mean(data):
    if len(data.shape) == 2:
        num, dim = data.shape
    elif len(data.shape) == 1:
        return data
    p = np.array([1, 0])
    while True:
        angle_sum = 0
        for index in range(num):
            x = data[index, :]
            angle = np.arccos(p @ x)
            # print(angle)
            if angle >= 0.01:
                x_tilde = (x - p * np.cos(angle)) * angle / np.sin(angle)
                angle_sum = angle_sum + x_tilde

        x_tilde = 1 / num * angle_sum
        if np.linalg.norm(x_tilde) <= 0.01:
            return np.array([p[0], p[1]])
        else:
            p = p * np.cos(np.linalg.norm(x_tilde)) + x_tilde / np.linalg.norm(x_tilde) * np.sin(np.linalg.norm(x_tilde))


def calc_z_value(data_tilde, data):
    x = karcher_mean(data)
    p = data_tilde
    angle  = np.arccos(p @ x)
    if angle != 0:
        x_tilde = (x - p * np.cos(angle)) * angle / np.sin(angle)
    else:
        x_tilde = np.zeros(2)
    # return np.linalg.norm(x_tilde)
    return np.exp(np.linalg.norm(x_tilde)-np.pi/2)**2


def gibbs_sampler(data, assignment_array, prior_distribution, alpha):
    pos_data = data[:, 0:2]

    (N, M) = pos_data.shape
    for i in np.random.permutation(N):
        data_i = data[i, :]
        data_no_i = np.delete(data, i, axis=0)
        assignment_array_no_i = np.delete(assignment_array, i)
        values, counts = np.unique(assignment_array_no_i, return_counts=True)
        K = values.shape[0]
        cond_prob = np.zeros(K + 1)

        for k, c_k in enumerate(values):
            data_c_k = data_no_i[assignment_array_no_i == c_k, :]
            z  = calc_z_value(data_i[2:4], data_c_k[:, 2:4])
            augmented_data_i = np.hstack((data_i[0:2], z))

            augmented_data_c_k = np.hstack((data_c_k[:, 0:2], np.zeros((data_c_k.shape[0], 1))))

            data_post_pred = prior_distribution.posterior_predictive(data_tilde=augmented_data_i,
                                                                     data=augmented_data_c_k)
            cond_prob[k] = counts[k] / (N - 1 + alpha) * data_post_pred

        cond_prob[-1] = alpha / (N - 1 + alpha) * prior_distribution.prior_predictive(data_tilde=augmented_data_i)

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

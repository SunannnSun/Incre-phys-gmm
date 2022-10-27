from gibbs_sampling import *
import random
from sklearn.mixture import BayesianGaussianMixture
import matplotlib as mpl
import itertools

from gibbs_sampling_2 import *
import scipy.io as scio
import matplotlib.pyplot as plt
import time
from load_dataset import load_dataset
from adaptive_kernel import trajectory_pre_processing
from spectral_clustering import spectral_clustering


def normalize_velocity(data):
    vel_data = data[2:4, :]
    vel_norm = np.linalg.norm(vel_data, axis=0)
    normalized_vel_data = np.divide(vel_data, vel_norm)
    new_data = data
    new_data[2:4, :] = normalized_vel_data
    return new_data.T


if __name__ == "__main__":
    """Initialize Data"""
    pkg_dir = 'datasets/'
    chosen_dataset = 10
    sub_sample = 4  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
    nb_trajectories = 7  # For real 3D data
    Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
    Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
    Data = normalize_velocity(Data)
    Xi_ref = Data[:, 0:2]
    Xi_dot_ref = Data[:, 2:4]
    # Introduce model complexity in the process of sampling#
    new_feature = np.zeros((Data.shape[0], 3))
    for i in range(Data.shape[0]):
        new_feature[i, 0] = np.linalg.norm(Data[i, :])**6
        new_feature[i, 1] = Data[i, 0] * Data[i, 2]
        # new_feature[i, 1] = Data[i, 0]**2
        new_feature[i, 2] = Data[i, 1] * Data[i, 3]
    Data = np.hstack((Data[:, 0:2], new_feature))

    # Data = Data[:, 0:2]

    """Initialize Prior"""
    lambda_0 = {
        "nu_0": 5,
        "Sigma_0": np.eye(2)
    }
    # kde = kde_prior(Xi_ref, Xi_dot_ref, option="full")

    prior = mixture_prior(Data)

    """Initial Guess of assignment_array and parameter_list"""
    init_option = 1
    # option 1: pre-partition of trajectory data
    # option 2: one cluster for each point

    if init_option == 1:
        # C_array, phi_list = trajectory_pre_processing(Xi_ref)
        C_array, phi_list = spectral_clustering(Xi_ref, Xi_dot_ref)
    else:
        phi_list = []
        for n in range(Data.shape[0]):
            # mu, sigma = prior.posterior_rvs(Data[n, :])
            sigma = draw_posterior_inverse_wishart(Data[n, 0:2][np.newaxis, :], lambda_0)
            # mu = draw_posterior_kde(Xi_ref[n, :], sigma, kde)
            mu = kde.posterior_rvs(Data[n, 0:2], sigma)
            phi_list.append((mu, sigma))
        C_array = np.arange(Data.shape[0])

    """Begin Gibbs Sampling"""
    print(Data.shape)
    for iteration in range(30):

        """"""
        # C_array, phi_list = gibbs_sampling(Data[:, 0:2], C_array, phi_list, kde, lambda_0, alpha=0.1, num_aux=2)
        # if iteration < 15:
        #     kde.update(C_array_pre, phi_list_pre)
        # kde.update(C_array, phi_list)
        """"""

        C_array = gibbs_sampling_2(Data, C_array, prior, alpha=0.001)

        print("Iteration: %d; Number of Components: %d" % ((iteration + 1), np.max(C_array) + 1))


    """Plot results"""
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]

    for n in range(Data.shape[0]):
        color = colors[C_array[n]]
        ax.scatter(Data[n, 0], Data[n, 1], c=color)
    plt.show()

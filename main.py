from gibbs_sampling import *
import random
from sklearn.mixture import BayesianGaussianMixture
import matplotlib as mpl
import itertools
import ds_tools.mousetrajectory_gui as mt

from gibbs_sampling_2 import *
from gibbs_sampling_3 import *

import scipy.io as scio
import matplotlib.pyplot as plt
import time
from load_dataset import load_dataset
from adaptive_kernel import trajectory_pre_processing
from spectral_clustering import spectral_clustering
from demo_drawData import *

def normalize_data(data):
    return np.divide(data - np.mean(data, axis=0), np.sqrt(np.diag(np.cov(data.T))))


def normalize_velocity(data):
    vel_data = data[2:4, :]
    vel_norm = np.linalg.norm(vel_data, axis=0)
    normalized_vel_data = np.divide(vel_data, vel_norm)
    new_data = data
    new_data[2:4, :] = normalized_vel_data
    return new_data.T


def dis_vectors(pos_data, if_normalize=True):
    num, dim = pos_data.shape
    dis_arr = np.zeros((num, dim))
    for index in range(pos_data.shape[0] - 1):
        dis = pos_data[index + 1, :] - pos_data[index, :]
        if index != 0 and np.linalg.norm(dis) > 5 * np.linalg.norm(dis_arr[index-1]):
            dis_arr[index] = dis_arr[index-1]
        else:
            dis_arr[index] = dis
    dis_arr[-1] = dis_arr[-2]
    if if_normalize:
        dis_norm_arr = np.linalg.norm(dis_arr, axis=1)
        dis_arr = np.divide(dis_arr, np.vstack((dis_norm_arr, dis_norm_arr)).T)
    # print(dis_arr)
    return dis_arr


def vel_vectors(t_coord, x_coord, y_coord):
    pos_data = np.hstack((np.array(x_coord).reshape(-1, 1), np.array(y_coord).reshape(-1, 1)))
    entry = np.ones((pos_data.shape[0]), dtype=int)
    for index in np.arange(1, len(t_coord)):
        if np.all(pos_data[index, :] == pos_data[index-1, :]):
            entry[index] = 0
    pos_data = pos_data[entry == 1, :]
    vel_data = np.zeros((pos_data.shape[0], 2))

    for index in np.arange(0, vel_data.shape[0]-1):
        vel_data[index, :] = (pos_data[index+1, :] - pos_data[index, :])
    vel_data[-1, :] = vel_data[-2, :]
    return np.hstack((pos_data, vel_data))


def dist_map(angle, max_height):
    slope = max_height / np.pi
    if angle <= 0:
        return slope * angle + slope * np.pi
    else:
        return -slope * angle + slope * np.pi


if __name__ == "__main__":
    """Import Data"""
    data_option = 2

    if data_option == 1:
        pkg_dir = 'datasets/'
        chosen_dataset = 8
        sub_sample = 3  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
        nb_trajectories = 7  # For real 3D data
        Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
        Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
        Data = normalize_velocity(Data)

    elif data_option == 2:
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        points, = ax.plot([], [], 'ro', markersize=2, lw=2)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.title('Draw trajectories to learn a motion policy:', fontsize=15)

        # Add UI buttons for data/figure manipulation
        store_btn = plt.axes([0.67, 0.05, 0.075, 0.05])
        clear_btn = plt.axes([0.78, 0.05, 0.075, 0.05])
        snap_btn = plt.axes([0.15, 0.05, 0.075, 0.05])
        bstore = Button(store_btn, 'store')
        bclear = Button(clear_btn, 'clear')
        bsnap = Button(snap_btn, 'snap')

        # Calling class to draw data on top of environment
        indexing = 2  # Set to 1 if you the snaps/data to be indexed with current time-stamp
        store_mat = 0  # Set to 1 if you want to store data in .mat structure for MATLAB
        draw_data = MouseTrajectory(points, indexing, store_mat)
        draw_data.connect()
        bstore.on_clicked(draw_data.store_callback)
        bclear.on_clicked(draw_data.clear_callback)
        bsnap.on_clicked(draw_data.snap_callback)

        # Show
        plt.show()
        file_name = './data/human_demonstrated_trajectories_1.dat'
        l, t, x, y = mt.load_trajectories(file_name)
        Data = np.hstack((np.array(x[2:-1]).reshape(-1, 1), np.array(y[2:-1]).reshape(-1, 1)))
        Data = vel_vectors(t[2:-1], x[2:-1], y[2:-1])
        Data = normalize_velocity(Data.T)




    """Plot results"""
    fig, ax = plt.subplots()
    ax.scatter(Data[:, 0], Data[:, 1], c='k')
    ax.set_aspect('equal')

    # Data = np.hstack((normalize_data(Data[:, 0:2]), Data[:, 2:4]))
    # Data[:, 0:2] = Data[:, 0:2] / 100
    # Data = Data.T

    # dis_array = dis_vectors(Data[:, 0:2])
    # Data = np.hstack((Data[:, 0:2], dis_array))

    # Introduce model complexity in the process of sampling #
    # new_feature = np.zeros((Data.shape[0], 1))
    # for i in range(Data.shape[0]):
    #     new_feature[i, 0] = np.arctan2(dis_array[i, 1], dis_array[i, 0])

        # new_feature[i, 0] = np.linalg.norm(Data[i, :])**2
        # new_feature[i, 0] = dist_map(np.arctan2(dis_array[i, 1], dis_array[i, 0]), 1)
        # new_feature[i, 1] = Data[i, 0] * Data[i, 2]
        # new_feature[i, 1] = Data[i, 0]**2
        # new_feature[i, 2] = Data[i, 1] * Data[i, 3]

    # Data = np.hstack((Data, new_feature))
    # Data = np.hstack((Data[:, 0:2], new_feature))
    # Data = Data[:, 0:2]

    """Initialize Prior"""
    # lambda_0 = {
    #     "nu_0": 5,
    #     "Sigma_0": np.eye(2)
    # }
    # kde = kde_prior(Xi_ref, Xi_dot_ref, option="full")

    prior = mixture_prior(Data)
    # prior = mixture_prior(Data[:, 0:2])

    """Initial Guess of assignment_array and parameter_list"""
    init_option = 1
    # option 1: pre-partition of trajectory data
    # option 2: one cluster for each point

    if init_option == 1:
        Xi_ref = Data[:, 0:2]
        Xi_dot_ref = Data[:, 2:4]
        C_array, phi_list = spectral_clustering(Xi_ref, Xi_dot_ref)
    elif init_option == 2:
        C_array = np.random.permutation(Data.shape[0])

    """Begin Gibbs Sampling"""
    print(Data.shape)
    # C_array = np.hstack((np.zeros(5), np.ones(80), 2 * np.ones(15)))
    # C_array = np.array(C_array, dtype=int)

    print("Iteration: %d; Number of Components: %d" % (0, np.max(C_array) + 1))
    for iteration in range(30):

        """"""
        # C_array, phi_list = gibbs_sampling(Data[:, 0:2], C_array, phi_list, kde, lambda_0, alpha=0.1, num_aux=2)
        # if iteration < 15:
        #     kde.update(C_array_pre, phi_list_pre)
        # kde.update(C_array, phi_list)
        """"""

        # C_array = gibbs_sampling_2(Data[:, 0:2], C_array, prior, alpha=0.1)
        C_array = gibbs_sampling_3(Data, C_array, prior, alpha=1)

        print("Iteration: %d; Number of Components: %d" % ((iteration + 1), np.max(C_array) + 1))


    """Plot results"""
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    for n in range(Data.shape[0]):
        color = colors[C_array[n]]
        ax.scatter(Data[n, 0], Data[n, 1], c=color)
    ax.set_aspect('equal')
    plt.show()

import random

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import ruptures as rpt
from load_dataset import load_dataset


def trajectory_pre_processing(data, one_prior=False, plot_flag=True):
    """Given sequential trajectory data, run an adaptive kernel through the data and return a preliminary partition of
    the data as the initial state of gibbs sampling"""

    (N, M) = data.shape
    print(N)
    assignment_array = np.zeros(N, dtype=int)
    parameter_list = []
    if one_prior:
        parameter_list.append((np.mean(data, axis=0), np.cov(data.T)))
        return assignment_array, parameter_list

    acc_angles = np.zeros(N - 2)
    for i in range(1, N - 2):
        disp_1 = data[i, :] - data[i - 1, :]
        disp_2 = data[i + 1, :] - data[i, :]
        acc_angles[i] = np.arctan2(np.linalg.det(np.array([disp_1, disp_2])), disp_1 @ disp_2)
    acc_angles = np.cumsum(acc_angles)

    """"""
    # acc_angles = np.convolve(acc_angles, np.ones(5) / 5, mode='same')
    # acc_angles = savgol_filter(acc_angles, 5, 2, mode='nearest')
    # acc_angles = moving_median(acc_angles, 20)
    x = np.arange(acc_angles.shape[0])
    algo = rpt.Pelt(model="linear").fit(np.column_stack((acc_angles.reshape(-1, 1), x)))
    """Human Choice!!!"""
    my_bkps = algo.predict(pen=20)
    my_bkps.pop(-1)
    # my_bkps = [45, 65]
    # my_bkps = []
    # """"""

    my_bkps.append(N)
    count = 0
    for i in range(N):
        if i < my_bkps[count]:
            assignment_array[i] = count
        else:
            count += 1
            assignment_array[i] = count
    # print(my_bkps)
    # print(assignment_array)

    for k in range(len(my_bkps)):
        mu_k = np.mean(data[assignment_array == k], axis=0)
        cov_k = np.cov(data[assignment_array == k].T)
        parameter_list.append((mu_k, cov_k))
    if plot_flag:
        fig, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].plot(acc_angles)
        axes[1].scatter(data[:, 0], data[:, 1], c="k")
        my_bkps.pop(-1)
        axes[0].plot(my_bkps, acc_angles[my_bkps], '*', c='r')
        axes[1].plot(data[my_bkps, 0], data[my_bkps, 1], '*', c='r')
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]
        for n in range(data.shape[0]):
            color = colors[assignment_array[n]]
            axes[2].scatter(data[n, 0], data[n, 1], c=color)

    # print(parameter_list[-1])
    return assignment_array, parameter_list


def moving_median(x, w):
    shifted = np.zeros((len(x)+w-1, w))
    shifted[:,:] = np.nan
    for idx in range(w-1):
        shifted[idx:-w+idx+1, idx] = x
    shifted[idx+1:, idx+1] = x
    # print(shifted)
    medians = np.median(shifted, axis=1)
    for idx in range(w-1):
        medians[idx] = np.median(shifted[idx, :idx+1])
        medians[-idx-1] = np.median(shifted[-idx-1, -idx-1:])
    return medians[(w-1)//2:-(w-1)//2]


if __name__ == "__main__":
    # Data = scio.loadmat('Increm_Learning/fork.mat')['Xi_ref'].T
    # Data = Data[np.arange(0, Data.shape[0], 2)]
    # Data = scio.loadmat('Increm_Learning/linearity.mat')['Xi_ref'].T
    # Data = Data[np.arange(0, Data.shape[0], 2)]
    # Data = Data[220:-1, :]


    pkg_dir = 'datasets/'
    chosen_dataset = 8
    sub_sample = 2  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
    nb_trajectories = 7  # For real 3D data
    Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
    Data = Data[0:2, :].T
    trajectory_pre_processing(Data)
    # print(np.convolve([1,2,3],[0.1,0.1, 0.1], 'same'))
    plt.show()

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import ruptures as rpt


def trajectory_pre_processing(data, plot_flag=False):
    """Given sequential trajectory data, run an adaptive kernel through the data and return a preliminary partition of
    the data as the initial state of gibbs sampling"""

    (N, M) = data.shape
    assignment_array = np.zeros(N, dtype=int)
    parameter_list = []

    acc_angles = np.zeros(N - 2)
    for i in range(1, N - 2):
        disp_1 = data[i, :] - data[i - 1, :]
        disp_2 = data[i + 1, :] - data[i, :]
        acc_angles[i] = np.arctan2(np.linalg.det(np.array([disp_1, disp_2])), disp_1 @ disp_2)

    acc_angles = np.cumsum(acc_angles)
    fig, ax = plt.subplots()
    ax.plot(acc_angles)
    # acc_angles = np.convolve(acc_angles, np.ones(10)/10, mode='same')
    # acc_angles = savgol_filter(acc_angles, 25, 2, mode='nearest')
    algo = rpt.Pelt(model="clinear").fit(acc_angles)
    my_bkps = algo.predict(pen=3)
    my_bkps.pop(-1)
    ax.plot(my_bkps, acc_angles[my_bkps], '*', c='r')
    my_bkps.append(N)
    count = 0
    for i in range(N):
        if i <= my_bkps[count]:
            assignment_array[i] = count
        else:
            count += 1
            assignment_array[i] = count
    print(my_bkps)

    for k in range(len(my_bkps)):
        mu_k = np.mean(data[assignment_array == k], axis=0)
        cov_k = np.cov(data[assignment_array == k].T)
        parameter_list.append((mu_k, cov_k))
    return assignment_array, parameter_list


if __name__ == "__main__":
    Data = scio.loadmat('Increm_Learning/double.mat')['Xi_ref'].T
    trajectory_pre_processing(Data)
    # print(np.convolve([1,2,3],[0.1,0.1, 0.1], 'same'))
    # plt.show()

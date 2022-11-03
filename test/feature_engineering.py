import random
from sklearn.mixture import BayesianGaussianMixture
from gibbs_sampling_single_prior import *
import matplotlib.pyplot as plt
from util.load_matlab_data import load_dataset



def plot_results(X, X_t, Y_, means, covariances, index, title):
    color_iter = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]
    splot = plt.figure()
    ax = plt.axes(projection='3d')
    fig, ax2 = plt.subplots()
    # fig, ax3 = plt.subplots()

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # v, w = np.linalg.eigh(covar)
        # v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        # u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        if X.shape[1] >= 3:
            ax.scatter3D(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ == i, -1], color=color)
        ax2.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=color)
        # ax3.scatter(X_t[Y_ == i, 0], X_t[Y_ == i, 1], color=color)

        # Plot an ellipse to show the Gaussian component
        # angle = np.arctan(u[1] / u[0])
        # angle = 180.0 * angle / np.pi  # convert to degrees
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(0.5)p
        # splot.add_artist(ell)
    ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])))

    plt.xticks(())
    plt.yticks(())
    plt.title(title)


def dis_vectors(pos_data, if_normalize=True):
    num, dim = pos_data.shape
    dis_arr = np.zeros((num, dim))
    for index in range(pos_data.shape[0] - 1):
        dis = pos_data[index + 1, :] - pos_data[index, :]
        if index != 0 and np.linalg.norm(dis) > 5 * np.linalg.norm(dis_arr[index - 1]):
            dis_arr[index] = dis_arr[index - 1]
        else:
            dis_arr[index] = dis
    dis_arr[-1] = dis_arr[-2]
    if if_normalize:
        dis_norm_arr = np.linalg.norm(dis_arr, axis=1)
        dis_arr = np.divide(dis_arr, np.vstack((dis_norm_arr, dis_norm_arr)).T)
    # print(dis_arr)
    return dis_arr


def dist_map(angle, max_height):
    slope = max_height / np.pi
    if angle <= 0:
        return slope * angle + slope * np.pi
    else:
        return -slope * angle + slope * np.pi



# ang = np.linspace(-np.pi, np.pi, 100)
# ang_dist = np.zeros((100,))
# for i in range(100):
#     ang_dist[i] = dist_map(ang[i], 100)
#
# fig, ax100 = plt.subplots()
# ax100.scatter(ang, ang_dist)





pkg_dir = 'matlab_data/'
chosen_dataset = 10
sub_sample = 4  # % '>2' for real 3D Datasets, '1' for 2D toy matlab_data
nb_trajectories = 7  # For real 3D data
Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
Data = normalize_velocity(Data)
# Data = Data.T

dis_array = dis_vectors(Data[:, 0:2])
# Data = np.hstack((Data[:, 0:2], dis_array))
# Data = Data[:, [0,1,2,3]]

new_feature = np.zeros((Data.shape[0], 1))
for i in range(Data.shape[0]):
    new_feature[i, 0] = np.linalg.norm(Data[i, :]) ** 2
    # new_feature[i, 0] = dist_map(np.arctan2(dis_array[i, 1] , dis_array[i, 0]), 5)
    # new_feature[i, 0] = dist_map(np.arctan2(Data[i, 3] , Data[i, 2]), 1)

    new_feature[i, 0] = np.arctan2(dis_array[i, 1] , dis_array[i, 0])
    # new_feature[i, 0] = np.arctan2(Data[i, 3] , Data[i, 2])

Data = np.hstack((Data[:, 0:2], new_feature))
# Data = np.hstack((Data, new_feature))

Data_t = Data

# if Data.shape[1] >= 3:
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(Data[:, 0], Data[:, 1], Data[:, 2], c=Data[:, 2], cmap='Greens')

dpgmm = BayesianGaussianMixture(n_components=100, covariance_type="full").fit(Data_t)

plot_results(Data_t, Data, dpgmm.predict(Data_t), dpgmm.means_, dpgmm.covariances_, 1,
             "Bayesian Gaussian Mixture with a Dirichlet process prior")
plt.show()

from gibbs_sampling import *
import random
from sklearn.mixture import BayesianGaussianMixture
import matplotlib as mpl
from gibbs_sampling_2 import *
import scipy.io as scio
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from main import normalize_velocity


def plot_results(X, Y_, means, covariances, index, title):
    color_iter = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]
    splot = plt.figure()
    ax = plt.axes(projection='3d')
    fig, ax2 = plt.subplots()
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

        # Plot an ellipse to show the Gaussian component
        # angle = np.arctan(u[1] / u[0])
        # angle = 180.0 * angle / np.pi  # convert to degrees
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(0.5)p
        # splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


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


pkg_dir = 'datasets/'
chosen_dataset = 8
sub_sample = 4  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 7  # For real 3D data
Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
Data = normalize_velocity(Data)
# Data = Data.T


dis_array = dis_vectors(Data[:, 0:2])
Data = np.hstack((Data[:, 0:2], dis_array))
# Data = Data[:, 0:2]

a = 0.2
new_feature = np.zeros((Data.shape[0], 1))
for i in range(Data.shape[0]):
    # new_feature[i, 0] = Data[i, 0] * Data[i, 1]
    # new_feature[i, 1] = Data[i, 0]**2
    # new_feature[i, 2] = Data[i, 1]**2
    new_feature[i, 0] = np.linalg.norm(Data[i, :]) ** 2
    # new_feature[i, 0] = np.sum(Data[i, :])
    # new_feature[i, 0] = np.arctan2(Data[i, 2], Data[i, 3])
Data = np.hstack((Data, new_feature))
print(Data.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(Data[:, 0], Data[:, 1], Data[:, -1], c=Data[:, 2], cmap='Greens')

dpgmm = BayesianGaussianMixture(n_components=10, covariance_type="full").fit(Data)

plot_results(
    Data,
    dpgmm.predict(Data),
    dpgmm.means_,
    dpgmm.covariances_,
    1,
    "Bayesian Gaussian Mixture with a Dirichlet process prior")
plt.show()

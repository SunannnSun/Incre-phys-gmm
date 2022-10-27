import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from load_dataset import load_dataset

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering


def compute_similarity_matrix(Xi_ref, Xi_dot_ref):
    estimate_l = 1
    length_scale = 0.5
    l_sensitivity = 2

    # if estimate_l == 1:
    #     D, mode_hist_D, mean_D = compute_dist(Xi_ref, 1)
    #     if mode_hist_D == 0:
    #         mode_hist_D = mean_D
    #     sigma = np.sqrt(mode_hist_D / l_sensitivity)  # warning because I haven't implemented full functionality
    #     l = 1 / (2 * (sigma ** 2))
    # else:
    l = length_scale
    # print('Computed length-scale l= {}'.format(l))

    # Compute element-wise cosine similarities
    len_of_Xi_dot = len(Xi_dot_ref[0])
    S = np.zeros((len_of_Xi_dot, len_of_Xi_dot))
    for i in np.arange(0, len_of_Xi_dot):
        for j in np.arange(0, len_of_Xi_dot):
            cos_angle = (np.dot(Xi_dot_ref[:, i], Xi_dot_ref[:, j])) / (
                    np.linalg.norm(Xi_dot_ref[:, i]) * np.linalg.norm(Xi_dot_ref[:, j]))
            if np.isnan(cos_angle):
                cos_angle = 0
            s = 1 + cos_angle

            # Compute Position component
            xi_i = Xi_ref[:, i]
            xi_j = Xi_ref[:, j]

            # Euclidean pairwise position-kernel
            p = np.exp(-l * np.linalg.norm(xi_i - xi_j))

            # Shifted Cosine Similarity of velocity vectors
            S[i][j] = p * s

        # Plot Similarity matrix
    title_str = 'Physically-Consistent Similarity Confusion Matrix'
    # sim_plot(S, title_str)
    return S


def sim_plot(S, title_str):
    fig, ax0 = plt.subplots(1, 1)
    c = ax0.pcolor(S, cmap='jet')
    ax0.set_title(title_str)
    fig.tight_layout()
    plt.colorbar(c)


def compute_dist(X, display_hist):
    X = np.transpose(X)
    print(np.shape(X))
    maxSamples = 10000
    if len(X) < maxSamples:
        X_train = X
        hist_distances = 1
    else:
        X_train = X[0:maxSamples, :]
        hist_distances = 10
    start = time.perf_counter()
    D = pdist(X_train, 'euclidean')
    end = time.perf_counter()
    mean_D = np.mean(D)
    max_D = np.max(D)
    hist, bin_edges = np.histogram(D, bins='auto')  # caution change bins to adjust result
    max_values_id = np.argmax(hist)
    print("pair calculations take: {}, mean is {}, max is {}".format(start - end, mean_D, max_D))
    return D, bin_edges[max_values_id], mean_D


def spectral_clustering(Xi_ref, Xi_dot_ref, if_plot=False):
    Xi_ref = Xi_ref.T
    Xi_dot_ref = Xi_dot_ref.T
    similarity_matrix = compute_similarity_matrix(Xi_ref, Xi_dot_ref)
    clustering = SpectralClustering(n_clusters=15,
                                    assign_labels='discretize', affinity="precomputed",
                                    random_state=0).fit(similarity_matrix)

    assignment_array = clustering.labels_
    # print(np.unique(assignment_array))
    parameter_list = []
    for k in np.unique(assignment_array):
        # print(k)
        mu_k = np.mean(Xi_ref[:, assignment_array == k], axis=1)
        cov_k = np.cov(Xi_ref[:, assignment_array == k])
        parameter_list.append((mu_k, cov_k))

    if if_plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]
        for n in range(Xi_ref.shape[1]):
            color = colors[assignment_array[n]]
            axes[0].scatter(Xi_dot_ref[0, n], Xi_dot_ref[1, n], c=color)
            axes[1].scatter(Xi_ref[0, n], Xi_ref[1, n], c=color)

    return assignment_array, parameter_list


if __name__ == "__main__":
    pkg_dir = 'datasets/'
    chosen_dataset = 10
    sub_sample = 10  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
    nb_trajectories = 7  # For real 3D data
    Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
    Data = Data[:, np.arange(0, Data.shape[1], 4)]
    Xi_ref = Data[0:2, :]
    Xi_dot_ref = Data[2:4, :]
    Xi_dot_ref_norm = np.linalg.norm(Xi_dot_ref, axis=0)
    Xi_dot_ref_normalized = np.divide(Xi_dot_ref, Xi_dot_ref_norm)
    # print(Xi_dot_ref_norm)
    # print(Xi_dot_ref_normalized)

    # print(Data.shape)
    # Xi_ref = Data[0:2, :]
    # Xi_dot_ref = Data[0:2, :]
    #
    # similarity_matrix = compute_similarity_matrix(Xi_ref, Xi_dot_ref)
    # print(similarity_matrix[0, 30:40])
    # import numpy as np
    #
    # X = np.array([[1, 1], [2, 1], [1, 0],
    #               [4, 7], [3, 5], [3, 6]])
    clustering = spectral_clustering(Xi_ref.T, Xi_dot_ref_normalized.T, True)

    # ax, fig = plt.subplots()
    # plt.scatter(Xi_dot_ref[0, :], Xi_ref[1, :], c='k')
    # clustering.labels_
    plt.show()

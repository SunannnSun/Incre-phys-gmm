import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def kernel_density_estimator(data, if_plot=False):
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=2)
    grid.fit(data)
    kde_object = grid.best_estimator_
    if if_plot:
        nx, ny = (50, 50)
        X, Y = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        pdf = np.exp(kde_object.score_samples(xy))
        Z = pdf.reshape(ny, nx)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # plt.show()
    return kde_object


if __name__ == "__main__":
    """2D KDE"""

    # np.random.seed(1)
    # N = 200
    # X1 = np.random.multivariate_normal([-2, -2], [[2, 1], [1, 1]], int(0.2 * N))
    # X2 = np.random.multivariate_normal([2, 2], [[2, 0], [0, 0.1]], int(0.8 * N))
    # X = np.concatenate((X1, X2))
    # print(X.shape)
    # kde = kernel_density_estimator(X)
    # print(kde.sample())
    # # Plot PDF of estimated distribution
    # nx, ny = (50, 50)
    # X, Y = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))
    # xy = np.vstack([X.ravel(), Y.ravel()]).T
    # pdf = np.exp(kde.score_samples(xy))
    # Z = pdf.reshape(ny, nx)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    # plt.show()



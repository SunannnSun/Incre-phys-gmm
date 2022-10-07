from gibbs_sampling import *
import scipy.io as scio
import matplotlib.pyplot as plt
import time

from adaptive_kernel import trajectory_pre_processing


"""Initialize Data"""
# np.random.seed(1)
# num_data = 40
# X1 = np.random.multivariate_normal([0, 0], [[15, 0.1], [0.1, 0.1]], int(0.3 * num_data))
# X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], int(0.7 * num_data))
# Data = np.concatenate((X1, X2))
# Data = X1
Data = scio.loadmat('Increm_Learning/linearity.mat')['Xi_ref'].T


"""Initialize Prior"""
lambda_0 = {
    "nu_0": 3,
    "Sigma_0": np.array([[1, 0], [0, 1]])
}
kde = kernel_density_estimator(Data, True)


"""Initial Guess of assignment_array and parameter_list"""
start = time.time()
init_option = 1
# option 1: pre-partition of trajectory data
# option 2: one cluster for each point

if init_option == 1:
    C_array, phi_list = trajectory_pre_processing(Data)
else:
    phi_list = []
    for n in range(Data.shape[0]):
        sigma = draw_posterior_inverse_wishart(Data[n, :][np.newaxis, :], lambda_0)
        mu = draw_posterior_kde(Data[n, :], sigma, kde)
        phi_list.append((mu, sigma))
    C_array = np.arange(Data.shape[0])

end = time.time()
print("Initialization Done. Time taken: %.2f sec" % (end - start))


"""Begin Gibbs Sampling"""
for iteration in range(100):
    C_array, phi_list = gibbs_sampling(Data, C_array, phi_list, kde, lambda_0)
    print("Iteration: %d; Number of Components: %d" % ((iteration+1), np.max(C_array)+1))


"""Plot results"""
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'w', 'y', 'crimson', 'lime']
for n in range(Data.shape[0]):
    color = colors[C_array[n]]
    ax.scatter(Data[n, 0], Data[n, 1], c=color)
plt.show()

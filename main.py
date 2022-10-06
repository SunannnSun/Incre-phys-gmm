from gibbs_sampling import *
import time


"""Initialize Data"""
np.random.seed(1)
num_data = 20
X1 = np.random.multivariate_normal([0, 0], [[15, 0.1], [0.1, 0.1]], int(0.3* num_data))
X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], int(0.7 * num_data))
Data = np.concatenate((X1, X2))
Data = X2
Data = scio.loadmat('Increm_Learning/S_shape.mat')['Xi_ref'].T

"""Initialize Prior"""
lambda_0 = {
    "nu_0": 3,
    "Sigma_0": np.array([[1, 0], [0, 1]])
}
kde = kernel_density_estimator(Data, True)

"""Initial Guess of assignment_array and parameter_list"""
start = time.time()
phi_list = []
for i in range(Data.shape[0]):
    sigma = draw_posterior_inverse_wishart(Data[i, :][np.newaxis, :], lambda_0)
    mu = draw_posterior_kde(Data[i, :], sigma, kde)
    phi_list.append((mu, sigma))
C_array = np.arange(Data.shape[0])
end = time.time()
print("Initialization Done. Time taken: %.2f sec" % (end - start))

"""Begin Gibbs Sampling"""
for iteration in range(50):
    C_array, phi_list = gibbs_sampling(Data, C_array, phi_list, kde, lambda_0)
    print("Iteration: %d; Number of Components: %d" % ((iteration+1), np.max(C_array)+1))

"""Plot results"""
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'w', 'y', 'crimson', 'lime']
for i in range(Data.shape[0]):
    color = colors[C_array[i]]
    ax.scatter(Data[i, 0], Data[i, 1], c=color)
plt.show()

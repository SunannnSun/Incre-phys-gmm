from main import *


pkg_dir = 'matlab_data/'
chosen_dataset = 10
sub_sample = 1 # % '>2' for real 3D Datasets, '1' for 2D toy matlab_data
nb_trajectories = 7  # For real 3D data
Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
Data = normalize_velocity(Data)
# print(Data.shape)
Data = np.vstack((Data[0:100, :], Data[300: 420, :]))
K = [0, 1, 2]
assignment_array = np.hstack((np.zeros(35), np.ones(40), 2*np.ones(25), np.zeros(90), 2*np.ones(30)))
ax = plt.axes(projection='3d')
ax.scatter3D(Data[:, 0], Data[:, 1], 0, c='k')

i = 40
data_i = Data[i, :]
Data = np.delete(Data, i, axis=0)
ax.scatter3D(data_i[0], data_i[1], 0, c='r')
assignment_array_i = np.delete(assignment_array, i)

for k in K:
    cluster_k = Data[assignment_array_i == k]
    # mean_angle = karcher_mean(cluster_k[:, 2:4])
    z = calc_z_value(data_i[2:4], cluster_k[:, 2:4])
    augmented_cluster_k = np.hstack((cluster_k, z * np.ones((cluster_k.shape[0], 1))))
    for i in range(cluster_k.shape[0]):
        ax.scatter3D(augmented_cluster_k[i, 0], augmented_cluster_k[i, 1], augmented_cluster_k[i, -1], c='b')



# aa = 0
# for i in range(35):
#     aa += np.rad2deg(np.arctan2(Data[i, 3], Data[i, 2]))
x1 = np.array([5, 5, 0])
x2 = np.array([5, 5, 1])

likelihood1 = multivariate_normal(mean=np.array([5, 6, 0]), cov=np.diag([1, 1, 1/2])).pdf(x1)
likelihood2 = multivariate_normal(mean=np.array([5, 5, 0]), cov=np.diag([1, 1, 1/2])).pdf(x2)

# likelihood2 = multivariate_normal(mean=np.array([0, 0, 0]), cov=np.array([[4, 0, 0], [0, 16, 0], [0, 0, np.pi]])).pdf(x2)

print(likelihood1)
print(likelihood2)

plt.show()
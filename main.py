from util.load_data import *
from util.process_data import *
from util.spectral_clustering import spectral_clustering
from prior_constructor import prior_class
from sampler import gibbs_sampler
import random


"""####### Load Data ############
option 1: draw and load data
option 2: load existing data
option 3: load matlab data ###"""

data_option = 2
if data_option == 1:
    draw_data()
    _, _, x, y = load_data()
    Data = add_directional_features(x, y, if_normalize=True)
elif data_option == 2:
    # data_name = 'human_demonstrated_trajectories_1.dat'
    data_name = 'stair.dat'
    _, _, x, y = load_data(data_name)
    Data = add_directional_features(x, y, if_normalize=True)
else:
    pkg_dir = 'data/'
    chosen_dataset = 8
    sub_sample = 4  # % '>2' for real 3D Datasets, '1' for 2D toy matlab_data
    nb_trajectories = 7  # For real 3D data
    Data = load_matlab_data(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
    Data = normalize_velocity_vector(Data)
    Data = Data.T
Data = Data[np.arange(0, Data.shape[0], 2), :]


"""#### Initialize Prior ####"""
prior = prior_class(Data)


""" Initialize Sampling Guess ###
option 1: single cluster ########
option 2: spectral clustering """
init_option = 1
if init_option == 1:
    C_array = np.random.permutation(Data.shape[0])
else:
    C_array, _ = spectral_clustering(Data[:, 0:2], Data[:, 2:4])

"""##### Begin Sampling ######"""
print("Data shape:", Data.shape)
for iteration in range(30):
    C_array = gibbs_sampler(Data, C_array, prior, alpha=1)
    print("Iteration: %d; Number of Components: %d" % ((iteration + 1), np.max(C_array) + 1))


"""##### Plot Results ######"""
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
for n in range(Data.shape[0]):
    color = colors[C_array[n]]
    ax.scatter(Data[n, 0], Data[n, 1], c=color)
ax.set_aspect('equal')
plt.show()

from main import *


def normalize_data(data):
    return np.divide(data - np.mean(data, axis=0), np.sqrt(np.diag(np.cov(data.T))))


pkg_dir = 'datasets/'
chosen_dataset = 10
sub_sample = 1  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 7  # For real 3D data
Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
Data = Data[:, np.arange(0, Data.shape[1], sub_sample)]  # (M by N)
Data = normalize_velocity(Data)
Data = np.vstack((Data[0:100, :], Data[300: 420, :]))

Data = Data - np.mean(Data, axis=0)
a = np.divide(Data, np.sqrt(np.diag(np.cov(Data.T))))
print(np.cov(a.T))

print(np.cov(normalize_data(Data).T))

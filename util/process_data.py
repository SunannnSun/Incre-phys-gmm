import numpy as np


def normalize_data(data):
    return np.divide(data - np.mean(data, axis=0), np.sqrt(np.diag(np.cov(data.T))))


def normalize_velocity_vector(data):
    vel_data = data[2:4, :]
    vel_norm = np.linalg.norm(vel_data, axis=0)
    normalized_vel_data = np.divide(vel_data, vel_norm)
    return np.hstack((data[0:2, :].T, normalized_vel_data.T))


def add_directional_features(x_coord, y_coord, if_normalize):
    pos_data = np.hstack((np.array(x_coord[1:-1]).reshape(-1, 1), np.array(y_coord[1:-1]).reshape(-1, 1)))
    entry = np.ones((pos_data.shape[0]), dtype=int)
    for index in np.arange(1, pos_data.shape[0]):
        if np.all(pos_data[index, :] == pos_data[index-1, :]):
            entry[index] = 0
    pos_data = pos_data[entry == 1, :]
    vel_data = np.zeros((pos_data.shape[0], 2))

    for index in np.arange(0, vel_data.shape[0]-1):
        vel_data[index, :] = (pos_data[index+1, :] - pos_data[index, :])
    vel_data[-1, :] = vel_data[-2, :]

    if if_normalize:
        vel_data_norm = np.linalg.norm(vel_data, axis=1)
        vel_data = np.divide(vel_data, np.hstack((vel_data_norm.reshape(-1, 1), vel_data_norm.reshape(-1, 1))))
    return np.hstack((pos_data, vel_data))


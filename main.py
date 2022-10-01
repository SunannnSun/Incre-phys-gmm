from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import scipy.io as scio


data = scio.loadmat('Increm_Learning/small_stair.mat')['Xi_ref'].T

print(data.shape)



import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
# For part a use 'data/points2D_Set1.mat'
file = h5py.File('data/points2D_Set1.mat', 'r')

# to flatten the data from multi dimension to 1D
X = file['x'][:].flatten()
Y = file['y'][:].flatten()
plt.scatter(X, Y)

slope, intercept = np.polyfit(X, Y, 1)

xvalues = np.array([X.min(), X.max()])
yvalues = slope * xvalues + intercept

# Overlaying the linear plot
plt.plot(line_x, line_y, color='yellow')

plt.show()

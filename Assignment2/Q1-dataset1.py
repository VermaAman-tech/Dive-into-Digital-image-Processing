import h5py
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, numpy as np
import numpy.random as rnd

# Load .mat file
# For part a use 'data/points2D_Set1.mat'
file = h5py.File('data/points2D_Set1.mat', 'r')

# Plot to show how points are scattered
#When I did x.shape it said its not a numpy array to I converted it to a numpy array and then took its transpose
X = np.array(file['x']).T
Y = np.array(file['y']).T
#Transpose because initial they were (1,300) and (1,300) on using without transpose
plt.scatter(X,Y)
plt.show()
X.shape
Y.shape
X_centered = X - np.mean(X)
Y_centered = Y - np.mean(Y)
print(np.mean(X))
X_centered.shape
print(np.mean(Y))
Y_centered.shape
covariance_matrix = np.cov(X_centered.T, Y_centered.T)
print("Covariance matrix ", covariance_matrix.shape, "\n")
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigen vectors ", eigenvectors.shape)
print("Eigen values ", eigenvalues.shape, "\n")
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
principal_component = sorted_eigenvectors[:, 0]
principal_component2 = sorted_eigenvectors[:, 1]
#principal_component
principal_component
principal_component2
X_centered.shape
import numpy as np
import matplotlib.pyplot as plt

# Define the point
point = np.array([np.mean(X), np.mean(Y)])

line_point = point - principal_component
plt.plot([point[0], line_point[0]], [point[1], line_point[1]], 'r-', label='Line PCA(useful)')
line_point = point + principal_component
plt.plot([point[0], line_point[0]], [point[1], line_point[1]], 'r-')
line_point = point - principal_component2
#To get my second line which is pc2 as mentioned in video
plt.plot([point[0], line_point[0]], [point[1], line_point[1]], 'r-', label='Line 2')
line_point = point + principal_component2
plt.plot([point[0], line_point[0]], [point[1], line_point[1]], 'r-')
plt.plot(point[0], point[1], 'bo', label='X_Mean,Y_Mean')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.scatter(X,Y)
plt.grid(True)
plt.show()

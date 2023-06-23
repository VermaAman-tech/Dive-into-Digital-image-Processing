import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('data/mnist.mat', 'r')

# Coverting int to float as mentioned in question
images = np.array(file['digits_train']).astype(float)
labels = np.array(file['labels_train']).astype(int)

# Reshape labels, it's an important step to associate each image to the corresponding label 
labels = labels.reshape(-1)

# Rather than computing independently, I used a function 
def analysis(digit):
    digit_images = images[labels == digit]  # To get the images of a particular digit
    
    # Reshape digit_images to (N, 784)
    # Mentioned in the question to do so.
    digit_images = digit_images.reshape(digit_images.shape[0], -1)
    
    # Computing the mean of each digit
    mean = np.mean(digit_images, axis=0)
    
    # Computing the covariance matrix
    covariance = np.cov(digit_images.T)
    
    # Getting the eigenvalues and eigenvectors from the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    # Sorting them in descending order to pick the first one as principal 
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Compute principal mode of variation (principal eigenvector)
    principal_mode = sorted_eigenvectors[:, 0]
    principal_eigenvalue = sorted_eigenvalues[0]
    principal_eigenvector = sorted_eigenvectors[:, 0]
    #print("Principal Eigenvector:")
    #print(principal_eigenvector)

    print("Principal Eigenvalue:")
    print(principal_eigenvalue)
    
    # Plot eigenvalues
    plt.plot(sorted_eigenvalues)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues for Digit ' + str(digit))
    plt.show()
    
    # Display images
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    fig.suptitle('Principal Mode of Variation for Digit ' + str(digit))
    
    # Reconstruct image using principal eigenvalue and eigenvector
    image = digit_images[0]
    reconstructed_image = mean - np.sqrt(principal_eigenvalue) * principal_mode

    axes[0].imshow(np.real(np.reshape(reconstructed_image, (28, 28))).T, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Reconstructed Image-1')
    
    
    # Display mean image in the center
    axes[1].imshow(np.real(np.reshape(mean, (28, 28))).T, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Mean Image')

    # Reconstruct image using principal eigenvalue and eigenvector
    image = digit_images[0]
    reconstructed_image2 = mean + np.sqrt(principal_eigenvalue) * principal_mode

    axes[2].imshow(np.real(np.reshape(reconstructed_image2, (28, 28))).T, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Reconstructed Image-2')

    plt.show()


# Pass every digit to the analysis function to compute the mentioned things
for digit in range(10):
    analysis(digit)

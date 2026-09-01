import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

#Applying SVD for image compression

def compress_image (image_path, num_singular_values):
    #Converting to Grayscale
    image=Image.open(image_path).convert('L') 
    image_matrix=np.array(image)

    #performing singular value decomposition
    U, S, VT= np.linalg.svd(image_matrix, full_matrices=False)

    #Keeping only the top 'num_singular_values' singular values
    compressed_image = U[:, :num_singular_values] @ np.diag(S[:num_singular_values]) @ VT[:num_singular_values, :]

    
    return compressed_image.astype(np.uint8), image_matrix

#Function to display Images

def plot_compression(original, compressed, num_singular_values):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(compressed, cmap='gray')
    plt.title(f"Compressed Image ({num_singular_values} Singular Values)")
    plt.axis('off')

    plt.show()


#Running the script
parser = argparse.ArgumentParser(description="Compress a grayscale image using SVD")
parser.add_argument("--image", type=str, default="Images/asus.jpg",
                     help="Path to the input image (default: Images/asus.jpg)")
parser.add_argument("--k", type=int, default=100,
                     help="Number of singular values to keep (default: 100)")
args = parser.parse_args()

image_path = args.image
num_singular_values = args.k

compressed, original = compress_image(image_path, num_singular_values)
plot_compression(original, compressed, num_singular_values)

compressed_image= Image.fromarray(compressed.astype(np.uint8))
compressed_image.save("Images/compressed_output.jpg")




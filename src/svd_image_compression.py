import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def plot_compression(original, compressed, num_singular_vlaues):
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
image_path = r"C:\Users\racht\OneDrive\Desktop\SVD-Image-Compression\Images\asus.jpg" # Change this to your image
num_singular_values = 100

compressed, original = compress_image(image_path, num_singular_values)
plot_compression(original, compressed, num_singular_values)

compressed_image= Image.fromarray(compressed.astype(np.uint8))
compressed_image.save(r"C:\Users\racht\OneDrive\Desktop\SVD-Image-Compression\images\compressed_output.jpg")





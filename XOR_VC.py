import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

# Function to split the image into 3 shares
def create_3_shares(image):
    # Get image dimensions (height, width, and number of channels)
    h, w, c = image.shape
    
    # Initialize three empty shares
    share1 = np.zeros((h, w, c), dtype=np.uint8)
    share2 = np.zeros((h, w, c), dtype=np.uint8)
    share3 = np.zeros((h, w, c), dtype=np.uint8)
    
    # Iterate through the image pixels
    for i in range(h):
        for j in range(w):
            for k in range(c):  # For each color channel (RGB)
                # Random pixel values for share1 and share2
                share1[i, j, k] = random.randint(0, 255)
                share2[i, j, k] = random.randint(0, 255)
                
                # Use XOR to create the third share
                share3[i, j, k] = image[i, j, k] ^ share1[i, j, k] ^ share2[i, j, k]
    
    return share1, share2, share3

# Function to reconstruct the original image from the 3 shares
def reconstruct_image(share1, share2, share3):
    h, w, c = share1.shape
    
    # Initialize the reconstructed image
    reconstructed_image = np.zeros((h, w, c), dtype=np.uint8)
    
    # Iterate through the shares and combine them to reconstruct the image
    for i in range(h):
        for j in range(w):
            for k in range(c):  # For each color channel (RGB)
                # XOR the three shares to get the original pixel value
                reconstructed_image[i, j, k] = share1[i, j, k] ^ share2[i, j, k] ^ share3[i, j, k]
    
    return reconstructed_image

# Load the input image (make sure the image is loaded in a compatible format)
image = cv2.imread('images/Xray.jpg') #change the input image name
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting with matplotlib

# Create 3 shares from the input image
share1, share2, share3 = create_3_shares(image)

# Save the shares
output_folder = 'images/Xray' # change the output folder or rename
os.makedirs(output_folder, exist_ok=True)

# Save the 3 shares as separate images
cv2.imwrite(os.path.join(output_folder, 'share_1.png'), cv2.cvtColor(share1, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, 'share_2.png'), cv2.cvtColor(share2, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, 'share_3.png'), cv2.cvtColor(share3, cv2.COLOR_RGB2BGR))

# Reconstruct the image using the 3 shares
reconstructed_image = reconstruct_image(share1, share2, share3)

# Save the reconstructed image
cv2.imwrite(os.path.join(output_folder, 'reconstructed_image.png'), cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR))

# Plot the input image, the 3 shares, and the reconstructed image in a row
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(share1)
axes[1].set_title('Share 1')
axes[1].axis('off')

axes[2].imshow(share2)
axes[2].set_title('Share 2')
axes[2].axis('off')

axes[3].imshow(share3)
axes[3].set_title('Share 3')
axes[3].axis('off')

axes[4].imshow(reconstructed_image)
axes[4].set_title('Reconstructed Image')
axes[4].axis('off')

plt.tight_layout()
plt.show()

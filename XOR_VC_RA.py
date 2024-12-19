import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(img1, img2):
    assert img1.shape == img2.shape, "Input images must have the same dimensions."
    mse = np.mean((img1 - img2) ** 2)
    return mse

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')  # Images are identical
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Function to calculate Structural Similarity Index (SSIM)
def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_value = np.mean(ssim_map)
    
    return ssim_value

# Function to display images and metrics in rows
def display_images_with_metrics(img1, shares, reconstructed, mse_values, psnr_values, ssim_values):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # First row: Display images
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')

    for i, share in enumerate(shares):
        axes[0, i + 1].imshow(cv2.cvtColor(share, cv2.COLOR_BGR2RGB))
        axes[0, i + 1].set_title(f"Share {i+1}")
        axes[0, i + 1].axis('off')

    axes[0, 4].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    axes[0, 4].set_title("Reconstructed Image")
    axes[0, 4].axis('off')

    # Second row: Empty first column, print MSE, PSNR, SSIM in the next columns
    axes[1, 0].axis('off')

    labels = ['MSE', 'PSNR', 'SSIM']
    
    # For each image (share and reconstructed), display the values
    for i in range(4):
        axes[1, i + 1].text(0.5, 0.5, f'{labels[0]}: {mse_values[i]:.2f}\n{labels[1]}: {psnr_values[i]:.2f} dB\n{labels[2]}: {ssim_values[i]:.4f}',
                            horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1, i + 1].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    # Load the original image
    img1 = cv2.imread('images/Xray.jpg')
    
    # Load the 3 shares
    share1 = cv2.imread('images/Xray/Share_1.png')
    share2 = cv2.imread('images/Xray/Share_2.png')
    share3 = cv2.imread('images/Xray/Share_3.png')
    
    # Load the reconstructed image
    reconstructed = cv2.imread('images/Xray/reconstructed_image.png')
    
    # Check if any image could not be loaded
    if img1 is None or share1 is None or share2 is None or share3 is None or reconstructed is None:
        print("Error: One or more images could not be loaded. Check the file paths.")
        return

    # Ensure all images are the same size
    if img1.shape != reconstructed.shape or img1.shape != share1.shape:
        print("Error: The images must have the same dimensions for analysis.")
        return

    # Calculate MSE, PSNR, SSIM for each share and the reconstructed image
    mse_values = [calculate_mse(img1, share1), calculate_mse(img1, share2), calculate_mse(img1, share3), calculate_mse(img1, reconstructed)]
    psnr_values = [calculate_psnr(img1, share1), calculate_psnr(img1, share2), calculate_psnr(img1, share3), calculate_psnr(img1, reconstructed)]
    ssim_values = [calculate_ssim(img1, share1), calculate_ssim(img1, share2), calculate_ssim(img1, share3), calculate_ssim(img1, reconstructed)]

    # Display images and metrics in rows
    display_images_with_metrics(img1, [share1, share2, share3], reconstructed, mse_values, psnr_values, ssim_values)

if __name__ == "__main__":
    main()

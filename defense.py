import torch
import torch.nn.functional as F

# Defense 1: Feature Squeezing.
def feature_squeeze(images, bit_depth = 4):
    """
    Reduces the number of bits used to represent each pixel.
    For example, if bit_depth is 4, then pixel values are quantized to 16 levels.
    """
    images = images.clone()
    levels = 2 ** bit_depth - 1  # e.g., 15 for 4 bits (0 to 15)
    images = torch.round(images * levels) / levels
    return images

# Defense 2: Artifact Defense via Gaussian Smoothing.
def gaussian_blur(images: torch.Tensor, device: torch.device, kernel_size: int = 3, sigma: float = 1.0):
    """
    Applies a Gaussian blur to each image to remove high-frequency adversarial artifacts.
    """
    channels = images.shape[1]
    # Create a 2D Gaussian kernel.
    grid = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    x, y = torch.meshgrid(grid, grid, indexing='ij')
    # Compute the squared Euclidean distance between x and y
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1).to(device)
    padding = kernel_size // 2
    # Apply the kernel to each image (using groups=channels for separate convolution on each channel)
    images_blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    return images_blurred

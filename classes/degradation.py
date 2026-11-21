import numpy as np

def apply_label_noise(labels: np.ndarray, p: float) -> np.ndarray:
    """
    Applies uniform label noise to a subset of labels.
    With probability p, each label is changed to a different random class.
    
    Args:
        labels: np.ndarray of shape (N,) or (N, 1) containing integer class labels.
        p: float (0.0 < p <= 1.0) representing the probability of corruption.
        
    Returns:
        np.ndarray containing the degraded labels.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Noise probability p must be in (0.0, 1.0]. Got: {p}")
    
    new_labels = labels.copy()
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    if n_classes <= 1:
        return new_labels
    
    # Determine the shape to correctly sample
    flat_labels = new_labels.flatten()
    N = len(flat_labels)
    
    # Create mask of which examples to corrupt
    corrupt_mask = np.random.rand(N) < p
    
    # For corrupted entries, choose a random label from the OTHER classes
    for i in np.where(corrupt_mask)[0]:
        current_label = flat_labels[i]
        other_classes = [c for c in unique_classes if c != current_label]
        flat_labels[i] = np.random.choice(other_classes)
        
    if new_labels.ndim > 1:
        new_labels = flat_labels.reshape(new_labels.shape)
    else:
        new_labels = flat_labels
        
    return new_labels

def apply_gaussian_blur(images: np.ndarray, kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Applies a 2D Gaussian blur to a set of images using discrete convolution.
    
    Args:
        images: np.ndarray of shape (N, H, W) or (N, H, W, C)
        kernel_size: int, odd, >= 3
        sigma: float, the standard deviation of the Gaussian distribution
        
    Returns:
        np.ndarray containing the blurred images.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be an odd integer >= 3. Got: {kernel_size}")
    
    # Generate 1D coordinate vector centered at the origin
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    # Construct 2D coordinate matrices for vectorized computation
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate the discretized 2D Gaussian kernel
    # The normalization constant 1/(2*pi*sigma^2) is omitted here as it 
    # cancels out during the subsequent sum-normalization step.
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    # Normalize the kernel so that the sum of all elements equals 1.0
    kernel = kernel / np.sum(kernel)
    
    
    
    blurred_images = np.zeros_like(images, dtype=float)
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2
    
    if images.ndim == 3:  # Grayscale (N, H, W)
        for i in range(images.shape[0]):
            img = images[i]
            padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            
            new_img = np.zeros_like(img, dtype=float)
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    region = padded_img[y:y+kernel_size, x:x+kernel_size]
                    new_img[y, x] = np.sum(region * kernel)
            blurred_images[i] = new_img
            
    elif images.ndim == 4:  # Color (N, H, W, C)
        for i in range(images.shape[0]):
            img = images[i]
            padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
            
            new_img = np.zeros_like(img, dtype=float)
            for c in range(img.shape[2]):
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        region = padded_img[y:y+kernel_size, x:x+kernel_size, c]
                        new_img[y, x, c] = np.sum(region * kernel)
            blurred_images[i] = new_img
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")
        
    if np.issubdtype(images.dtype, np.integer):
        # Round instead of truncating so constant regions are preserved exactly
        blurred_images = np.rint(blurred_images)
    return blurred_images.astype(images.dtype)

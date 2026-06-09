import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pywt
import os

# ── Noise addition ──────────────────────────────────────────────────────────

def add_gaussian_noise(img, sigma=30):
    noisy = img.astype(np.float64) + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255)

def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy().astype(np.float64)
    mask = np.random.random(img.shape[:2])
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy

# ── Metrics ─────────────────────────────────────────────────────────────────

def psnr(original, denoised):
    mse = np.mean((original.astype(np.float64) - denoised.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)

def ssim(img1, img2):
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x = x.std() ** 2
    sigma_y = y.std() ** 2
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return num / den

# ── Method 1: Gaussian filter (spatial) ─────────────────────────────────────

def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def convolve2d(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(ih):
        for j in range(iw):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out

def gaussian_denoise(img, sigma=2.0, size=7):
    k = gaussian_kernel(size, sigma)
    if img.ndim == 3:
        return np.stack([convolve2d(img[:,:,c], k) for c in range(3)], axis=2)
    return convolve2d(img, k)

# ── Method 2: Bilateral filter (edge-preserving) ─────────────────────────────

def bilateral_denoise(img, sigma_s=5, sigma_r=30, size=9):
    pad = size // 2
    if img.ndim == 3:
        return np.stack([_bilateral_channel(img[:,:,c], sigma_s, sigma_r, size, pad)
                         for c in range(3)], axis=2)
    return _bilateral_channel(img, sigma_s, sigma_r, size, pad)

def _bilateral_channel(channel, sigma_s, sigma_r, size, pad):
    padded = np.pad(channel.astype(np.float64), pad, mode='reflect')
    out = np.zeros_like(channel, dtype=np.float64)
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_w = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+size, j:j+size]
            center = padded[i+pad, j+pad]
            range_w = np.exp(-((patch - center)**2) / (2 * sigma_r**2))
            weights = spatial_w * range_w
            out[i, j] = (weights * patch).sum() / weights.sum()
    return out

# ── Method 3: Fourier low-pass filter ────────────────────────────────────────

def fourier_denoise(img, cutoff_ratio=0.1):
    if img.ndim == 3:
        return np.stack([_fourier_channel(img[:,:,c], cutoff_ratio)
                         for c in range(3)], axis=2)
    return _fourier_channel(img, cutoff_ratio)

def _fourier_channel(channel, cutoff_ratio):
    F = fftshift(fft2(channel.astype(np.float64)))
    h, w = channel.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * cutoff_ratio)
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy)**2 + (xx - cx)**2) <= radius**2
    F_filtered = F * mask
    result = np.real(ifft2(ifftshift(F_filtered)))
    return np.clip(result, 0, 255)

# ── Method 4: Wavelet thresholding ──────────────────────────────────────────

def wavelet_denoise(img, wavelet='db4', level=3, mode='soft'):
    if img.ndim == 3:
        return np.stack([_wavelet_channel(img[:,:,c], wavelet, level, mode)
                         for c in range(3)], axis=2)
    return _wavelet_channel(img, wavelet, level, mode)

def _wavelet_channel(channel, wavelet, level, mode):
    coeffs = pywt.wavedec2(channel.astype(np.float64), wavelet, level=level)
    detail_coeffs = [c for subband in coeffs[1:] for c in subband]
    all_details = np.concatenate([d.ravel() for d in detail_coeffs])
    sigma_est = np.median(np.abs(all_details)) / 0.6745
    threshold = sigma_est * np.sqrt(2 * np.log(channel.size))
    new_coeffs = [coeffs[0]]
    for subband in coeffs[1:]:
        new_subband = tuple(pywt.threshold(c, threshold, mode=mode) for c in subband)
        new_coeffs.append(new_subband)
    result = pywt.waverec2(new_coeffs, wavelet)
    result = result[:channel.shape[0], :channel.shape[1]]
    return np.clip(result, 0, 255)

# ── Method 5: Median filter ──────────────────────────────────────────────────

def median_denoise(img, size=3):
    if img.ndim == 3:
        return np.stack([_median_channel(img[:,:,c], size) for c in range(3)], axis=2)
    return _median_channel(img, size)

def _median_channel(channel, size):
    pad = size // 2
    padded = np.pad(channel.astype(np.float64), pad, mode='reflect')
    out = np.zeros_like(channel, dtype=np.float64)
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            out[i, j] = np.median(padded[i:i+size, j:j+size])
    return out

# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_all(original, noisy, results):
    print(f"\n{'Method':<20} {'PSNR (noisy)':<15} {'PSNR (denoised)':<18} {'SSIM (denoised)'}")
    print("-" * 68)
    orig_gray = np.mean(original, axis=2) if original.ndim == 3 else original
    noisy_gray = np.mean(noisy, axis=2) if noisy.ndim == 3 else noisy
    print(f"{'Noisy input':<20} {psnr(orig_gray, noisy_gray):<15.2f} {'-':<18} {'-'}")
    for name, denoised in results.items():
        d_gray = np.mean(denoised, axis=2) if denoised.ndim == 3 else denoised
        p = psnr(orig_gray, d_gray)
        s = ssim(orig_gray, d_gray)
        print(f"{name:<20} {'-':<15} {p:<18.2f} {s:.4f}")

# ── Visualization ────────────────────────────────────────────────────────────

def save_images(noisy, results, noise_label, output_dir, image_name="image"):
    os.makedirs(output_dir, exist_ok=True)
    def _save(arr, name):
        path = os.path.join(output_dir, name)
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
        print(f"Saved: {path}")
    _save(noisy, f"{image_name}_noisy_{noise_label}.png")
    for method_name, denoised in results.items():
        filename = f"{image_name}_denoised_{noise_label}_{method_name.lower().replace(' ', '_')}.png"
        _save(denoised, filename)

def plot_results(original, noisy, results, noise_type, save_path=None, show=True):
    items = [("Original", None), (f"Noisy ({noise_type})", None)] + list(results.items())
    n = len(items)
    cols = (n + 1) // 2
    fig = plt.figure(figsize=(4 * cols, 8))
    gs = gridspec.GridSpec(2, cols, figure=fig)

    def show(ax, img, title, psnr_val=None):
        ax.imshow(img.astype(np.uint8) if img.ndim == 3 else img.astype(np.uint8), cmap='gray')
        label = title if psnr_val is None else f"{title}\nPSNR: {psnr_val:.2f} dB"
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    orig_gray = np.mean(original, axis=2) if original.ndim == 3 else original
    noisy_gray = np.mean(noisy, axis=2) if noisy.ndim == 3 else noisy

    for idx, (title, img_data) in enumerate(items):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        if img_data is None:
            ref = original if title == "Original" else noisy
            p = None if title == "Original" else psnr(orig_gray, noisy_gray)
            show(ax, ref, title, p)
        else:
            d_gray = np.mean(img_data, axis=2) if img_data.ndim == 3 else img_data
            show(ax, img_data, title, psnr(orig_gray, d_gray))

    plt.tight_layout()
    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_fourier_spectrum(img, noisy, denoised_fourier, save_path=None, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original spectrum", "Noisy spectrum", "After low-pass filter"]
    images = [img, noisy, denoised_fourier]
    for ax, im, title in zip(axes, images, titles):
        gray = np.mean(im, axis=2) if im.ndim == 3 else im
        F = np.log(1 + np.abs(fftshift(fft2(gray))))
        ax.imshow(F, cmap='inferno')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_wavelet_coefficients(img, save_path=None, show=True):
    gray = np.mean(img, axis=2).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    coeffs = pywt.wavedec2(gray, 'db4', level=2)
    arr, slices = pywt.coeffs_to_array(coeffs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(np.abs(arr)), cmap='viridis')
    ax.set_title("Wavelet coefficient map (db4, 2 levels)", fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
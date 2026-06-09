import numpy as np
from PIL import Image
import os
import sys
from denoise import (
    add_gaussian_noise, add_salt_pepper_noise,
    gaussian_denoise, bilateral_denoise, fourier_denoise,
    wavelet_denoise, median_denoise,
    evaluate_all, plot_results, save_images,
    plot_fourier_spectrum, plot_wavelet_coefficients
)

# ── Config ───────────────────────────────────────────────────────────────────

args = sys.argv[1:]
SHOW_PLOTS = "--no-show" not in args
image_args = [a for a in args if not a.startswith("--")]
IMAGE_PATH = image_args[0] if image_args else "images/mountain.png"
image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAUSSIAN_SIGMA = 30
SP_PROB = 0.05

# ── Load image ───────────────────────────────────────────────────────────────

img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
print(f"Loaded: {IMAGE_PATH}  shape={img.shape}")

# ── Gaussian noise experiment ─────────────────────────────────────────────────

print("\n=== Gaussian Noise ===")
noisy_g = add_gaussian_noise(img, sigma=GAUSSIAN_SIGMA)

results_g = {
    "Gaussian filter": gaussian_denoise(noisy_g, sigma=2.0),
    "Bilateral filter": bilateral_denoise(noisy_g, sigma_s=5, sigma_r=30),
    "Fourier LP": fourier_denoise(noisy_g, cutoff_ratio=0.15),
    "Wavelet (soft)": wavelet_denoise(noisy_g, wavelet='db4', level=3, mode='soft'),
    "Median filter": median_denoise(noisy_g, size=3),
}

evaluate_all(img, noisy_g, results_g)
save_images(noisy_g, results_g, "gaussian", OUTPUT_DIR, image_name)
plot_results(img, noisy_g, results_g, "Gaussian σ=30",
             save_path=f"{OUTPUT_DIR}/{image_name}_gaussian_noise_results.png", show=SHOW_PLOTS)

plot_fourier_spectrum(img, noisy_g, results_g["Fourier LP"],
                      save_path=f"{OUTPUT_DIR}/{image_name}_fourier_spectrum.png", show=SHOW_PLOTS)

plot_wavelet_coefficients(img, save_path=f"{OUTPUT_DIR}/{image_name}_wavelet_coefficients.png", show=SHOW_PLOTS)

# ── Salt-and-pepper noise experiment ─────────────────────────────────────────

print("\n=== Salt-and-Pepper Noise ===")
noisy_sp = add_salt_pepper_noise(img, prob=SP_PROB)

results_sp = {
    "Gaussian filter": gaussian_denoise(noisy_sp, sigma=2.0),
    "Bilateral filter": bilateral_denoise(noisy_sp, sigma_s=5, sigma_r=50),
    "Fourier LP": fourier_denoise(noisy_sp, cutoff_ratio=0.15),
    "Wavelet (soft)": wavelet_denoise(noisy_sp, wavelet='db4', level=3, mode='soft'),
    "Median filter": median_denoise(noisy_sp, size=3),
}

evaluate_all(img, noisy_sp, results_sp)
save_images(noisy_sp, results_sp, "saltpepper", OUTPUT_DIR, image_name)
plot_results(img, noisy_sp, results_sp, f"Salt & Pepper p={SP_PROB}",
            save_path=f"{OUTPUT_DIR}/{image_name}_saltpepper_noise_results.png", show=SHOW_PLOTS)

# ── Parameter sweep: bilateral sigma_r ────────────────────────────────────────

print("\n=== Bilateral filter parameter sweep (sigma_r) ===")
sigma_r_values = [10, 30, 60, 100]
print(f"{'sigma_r':<12} {'PSNR':<10} {'SSIM'}")
print("-" * 30)

from denoise import psnr, ssim
orig_gray = np.mean(img, axis=2)

for sr in sigma_r_values:
    denoised = bilateral_denoise(noisy_g, sigma_s=5, sigma_r=sr)
    d_gray = np.mean(denoised, axis=2)
    p = psnr(orig_gray, d_gray)
    s = ssim(orig_gray, d_gray)
    print(f"{sr:<12} {p:<10.2f} {s:.4f}")

# ── Parameter sweep: wavelet levels ──────────────────────────────────────────

print("\n=== Wavelet decomposition level sweep ===")
print(f"{'Level':<10} {'PSNR':<10} {'SSIM'}")
print("-" * 28)

for level in [1, 2, 3, 4]:
    denoised = wavelet_denoise(noisy_g, wavelet='db4', level=level, mode='soft')
    d_gray = np.mean(denoised, axis=2)
    p = psnr(orig_gray, d_gray)
    s = ssim(orig_gray, d_gray)
    print(f"{level:<10} {p:<10.2f} {s:.4f}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
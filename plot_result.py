# --- START OF plot_results.py ---
import argparse
import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Attempt to import skimage for metrics in plot titles
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not installed. PSNR/SSIM in plot titles will not be available.")
    print("To install: pip install scikit-image")


def plot_image_results(lr_folder, sr_folder, hr_folder=None, num_images=5, output_plot_file=None):
    lr_paths = sorted(glob.glob(os.path.join(lr_folder, '*')))
    sr_paths_all = sorted(glob.glob(os.path.join(sr_folder, '*')))
    
    if not lr_paths:
        print(f"No images found in LR folder: {lr_folder}")
        return
    if not sr_paths_all:
        print(f"No images found in SR folder: {sr_folder}")
        return

    lr_basenames_map = {os.path.splitext(os.path.basename(p))[0]: p for p in lr_paths}
    
    images_to_plot = []
    plotted_lr_basenames = set()

    # Match SR images to LR images
    # Assumes SR filenames are like <lr_basename>_suffix.ext or <lr_basename>.ext
    for sr_path in sr_paths_all:
        if len(images_to_plot) >= num_images:
            break
        
        sr_basename_full, sr_ext = os.path.splitext(os.path.basename(sr_path))
        
        matched_lr_basename = None
        # Try to find the corresponding LR image
        # This logic assumes SR name is LR_basename + optional_suffix
        for lr_b in lr_basenames_map.keys():
            if sr_basename_full.startswith(lr_b):
                # Check if this lr_b is the longest prefix match to avoid ambiguity
                # e.g. lr_b="img", sr_basename_full="img_x_out"
                # if another lr_b="img_x", that one should match "img_x_out"
                if matched_lr_basename is None or len(lr_b) > len(matched_lr_basename):
                    matched_lr_basename = lr_b
        
        if matched_lr_basename and matched_lr_basename not in plotted_lr_basenames:
            lr_path = lr_basenames_map[matched_lr_basename]
            
            img_lr = cv2.imread(lr_path)
            img_sr = cv2.imread(sr_path)

            if img_lr is None or img_sr is None:
                print(f"Warning: Could not read LR ({lr_path}) or SR ({sr_path}) image. Skipping pair.")
                continue
            
            current_set = {'lr': img_lr, 'sr': img_sr, 'name': matched_lr_basename}

            if hr_folder:
                hr_path_found = None
                # Use original LR extension first, then common ones
                lr_original_ext = os.path.splitext(os.path.basename(lr_path))[1]
                possible_hr_extensions = [lr_original_ext, '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                # Remove duplicates and ensure lr_original_ext is first if not empty
                unique_exts = []
                if lr_original_ext: unique_exts.append(lr_original_ext)
                for ext in possible_hr_extensions:
                    if ext and ext not in unique_exts: unique_exts.append(ext)
                
                for ext_try in unique_exts:
                    hr_path_candidate = os.path.join(hr_folder, f"{matched_lr_basename}{ext_try}")
                    if os.path.isfile(hr_path_candidate):
                        hr_path_found = hr_path_candidate
                        break
                
                if hr_path_found:
                    img_hr = cv2.imread(hr_path_found)
                    if img_hr is not None:
                        current_set['hr'] = img_hr
                    else:
                        print(f"Warning: Could not read HR image: {hr_path_found}")
                else:
                    print(f"Note: HR image for {matched_lr_basename} not found in {hr_folder}.")
            
            images_to_plot.append(current_set)
            plotted_lr_basenames.add(matched_lr_basename)

    if not images_to_plot:
        print("No image pairs to plot after matching.")
        return

    num_cols = 2
    if hr_folder and any('hr' in img_set for img_set in images_to_plot):
        num_cols = 3
        
    num_rows = len(images_to_plot)
    fig_height_per_row = 5 
    plt.figure(figsize=(num_cols * 5, num_rows * fig_height_per_row))

    for i, img_set in enumerate(images_to_plot):
        # LR Image
        ax_lr = plt.subplot(num_rows, num_cols, i * num_cols + 1)
        ax_lr.imshow(cv2.cvtColor(img_set['lr'], cv2.COLOR_BGR2RGB))
        ax_lr.set_title(f"{img_set['name']} (LR)")
        ax_lr.axis('off')

        # SR Image
        ax_sr = plt.subplot(num_rows, num_cols, i * num_cols + 2)
        ax_sr.imshow(cv2.cvtColor(img_set['sr'], cv2.COLOR_BGR2RGB))
        title_sr = f"{img_set['name']} (SR)"
        
        if SKIMAGE_AVAILABLE and 'hr' in img_set:
            img_hr_metric = img_set['hr']
            img_sr_metric = img_set['sr']

            # Prepare for metrics (RGB, same shape)
            img_sr_metric_rgb = img_sr_metric[:, :, :3] if len(img_sr_metric.shape) == 3 and img_sr_metric.shape[2] == 4 else img_sr_metric
            img_hr_metric_rgb = img_hr_metric[:, :, :3] if len(img_hr_metric.shape) == 3 and img_hr_metric.shape[2] == 4 else img_hr_metric
            
            if img_sr_metric_rgb.shape != img_hr_metric_rgb.shape:
                img_hr_metric_rgb = cv2.resize(img_hr_metric_rgb, (img_sr_metric_rgb.shape[1], img_sr_metric_rgb.shape[0]), interpolation=cv2.INTER_AREA)

            if len(img_sr_metric_rgb.shape) == 2: img_sr_metric_rgb = cv2.cvtColor(img_sr_metric_rgb, cv2.COLOR_GRAY2BGR)
            if len(img_hr_metric_rgb.shape) == 2: img_hr_metric_rgb = cv2.cvtColor(img_hr_metric_rgb, cv2.COLOR_GRAY2BGR)

            if img_sr_metric_rgb.ndim == 3 and img_hr_metric_rgb.ndim == 3 and img_sr_metric_rgb.shape[2] == 3 and img_hr_metric_rgb.shape[2] == 3:
                try:
                    psnr = peak_signal_noise_ratio(img_hr_metric_rgb, img_sr_metric_rgb, data_range=255)
                    
                    win_size = min(7, min(img_sr_metric_rgb.shape[0], img_sr_metric_rgb.shape[1]))
                    if win_size % 2 == 0: win_size -=1
                    
                    ssim_val = np.nan
                    if win_size >= 3:
                        import skimage # Re-check version inside as it's conditional
                        skimage_version = list(map(int, skimage.__version__.split('.')))
                        if skimage_version[0] == 0 and skimage_version[1] < 19:
                             ssim_val = structural_similarity(img_hr_metric_rgb, img_sr_metric_rgb, data_range=255, multichannel=True, win_size=win_size)
                        else:
                             ssim_val = structural_similarity(img_hr_metric_rgb, img_sr_metric_rgb, data_range=255, channel_axis=-1, win_size=win_size)
                        title_sr += f"\nPSNR: {psnr:.2f} dB, SSIM: {ssim_val:.3f}"
                    else:
                        title_sr += f"\nPSNR: {psnr:.2f} dB (SSIM N/A: small)"
                except Exception as e:
                    print(f"Plot title metrics error for {img_set['name']}: {e}")
                    title_sr += "\n(Metrics error)"
        ax_sr.set_title(title_sr)
        ax_sr.axis('off')

        # HR Image (if available and num_cols is 3)
        if num_cols == 3 and 'hr' in img_set:
            ax_hr = plt.subplot(num_rows, num_cols, i * num_cols + 3)
            ax_hr.imshow(cv2.cvtColor(img_set['hr'], cv2.COLOR_BGR2RGB))
            ax_hr.set_title(f"{img_set['name']} (HR)")
            ax_hr.axis('off')

    plt.tight_layout()
    if output_plot_file:
        plt.savefig(output_plot_file)
        print(f"Plot saved to {output_plot_file}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot LR, SR, and optionally HR image results.")
    parser.add_argument('--lr_folder', type=str, required=True, help='Folder containing Low-Resolution input images.')
    parser.add_argument('--sr_folder', type=str, required=True, help='Folder containing Super-Resolved output images.')
    parser.add_argument('--hr_folder', type=str, default=None, help='Optional: Folder containing High-Resolution ground truth images.')
    parser.add_argument('--num_images', type=int, default=3, help='Number of image sets to plot.')
    parser.add_argument('--output_plot_file', type=str, default=None, help='Optional: Path to save the combined plot image.')
    
    args = parser.parse_args()
    plot_image_results(args.lr_folder, args.sr_folder, args.hr_folder, args.num_images, args.output_plot_file)
# --- END OF plot_results.py ---
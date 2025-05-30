import argparse
import cv2
import glob
import os
import matplotlib.pyplot as plt

def find_sr_image(lr_img_basename_no_ext, sr_dir, sr_suffix, sr_ext_preference, lr_original_ext_no_dot):
    """
    Attempts to find the corresponding SR image.
    """
    sr_basename_core = f"{lr_img_basename_no_ext}{sr_suffix}"

    potential_extensions = []
    if sr_ext_preference == 'auto':
        if lr_original_ext_no_dot:
            potential_extensions.append(lr_original_ext_no_dot)
        potential_extensions.extend(['png', 'jpg', 'jpeg', 'bmp', 'tiff']) # Common ones
    else:
        potential_extensions.append(sr_ext_preference)
        if sr_ext_preference != 'png': potential_extensions.append('png')
        if sr_ext_preference != 'jpg': potential_extensions.append('jpg')

    for ext in list(dict.fromkeys(potential_extensions)): # Unique extensions, keep order
        sr_path = os.path.join(sr_dir, f"{sr_basename_core}.{ext}")
        if os.path.exists(sr_path):
            return sr_path
    
    glob_pattern = os.path.join(sr_dir, f"{sr_basename_core}.*")
    matches = glob.glob(glob_pattern)
    if matches:
        return matches[0]
    return None

def display_image_set(lr_img_cv, sr_img_cv, hr_img_cv, 
                       lr_filename, sr_filename, hr_filename):
    """
    Displays LR, SR, and optionally HR images using matplotlib.
    """
    def prep_for_plot(img_cv):
        if img_cv is None: return None
        if img_cv.ndim == 2: return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        if img_cv.shape[2] == 3: return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        if img_cv.shape[2] == 4: return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
        return img_cv

    lr_display = prep_for_plot(lr_img_cv)
    sr_display = prep_for_plot(sr_img_cv)
    hr_display = prep_for_plot(hr_img_cv)

    num_valid_images = sum(x is not None for x in [lr_display, sr_display, hr_display])
    
    # Adjust columns: 3 if HR is present and valid, otherwise 2 (LR and SR)
    # If only one image is valid, plot 1.
    # More robustly: count valid images to plot
    images_to_plot = []
    titles = []
    if lr_display is not None:
        images_to_plot.append(lr_display)
        titles.append(f'LR: {lr_filename}')
    if hr_display is not None: # Plot HR second if present
        images_to_plot.append(hr_display)
        titles.append(f'HR: {hr_filename}')
    if sr_display is not None:
        images_to_plot.append(sr_display)
        titles.append(f'SR: {sr_filename}')
    
    num_cols = len(images_to_plot)
    if num_cols == 0:
        print("  No valid images to display for this set.")
        return

    fig, axes = plt.subplots(1, num_cols, figsize=(7 * num_cols, 7))
    if num_cols == 1: # If only one image, axes is not a list
        axes = [axes] 

    for i in range(num_cols):
        axes[i].imshow(images_to_plot[i])
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show() # This is the command to display the plot.
    plt.close(fig) # Close the figure to free memory, important in loops.

def main():
    parser = argparse.ArgumentParser(description="Plot LR, SR, and optionally HR images inline in a notebook.")
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory containing Low-Resolution (LR) images.')
    parser.add_argument('--sr_dir', type=str, required=True, help='Directory containing Super-Resolved (SR) images.')
    parser.add_argument('--hr_dir', type=str, default=None, help='(Optional) Directory containing High-Resolution (HR) ground truth images.')
    parser.add_argument('--num_to_plot', type=int, default=3, help='Number of image sets to plot. 0 or negative for all.')
    parser.add_argument('--sr_suffix', type=str, default='_out', help='Suffix for SR images (e.g., if LR is img.png and SR is img_out.png, suffix is _out). Use "" for no suffix.')
    parser.add_argument('--sr_ext', type=str, default='auto', help='Expected extension for SR images (e.g., png, jpg). "auto" tries LR extension then common ones.')

    args = parser.parse_args()

    # --- Crucial for notebook environments ---
    # This script assumes the notebook environment is already configured for inline plotting
    # (e.g., via %matplotlib inline in a previous cell).

    if not os.path.isdir(args.lr_dir):
        print(f"Error: LR directory not found: {args.lr_dir}"); return
    if not os.path.isdir(args.sr_dir):
        print(f"Error: SR directory not found: {args.sr_dir}"); return
    if args.hr_dir and not os.path.isdir(args.hr_dir):
        print(f"Warning: HR directory specified but not found: {args.hr_dir}. HR images will not be plotted."); args.hr_dir = None

    lr_image_paths = sorted(glob.glob(os.path.join(args.lr_dir, '*')))
    if not lr_image_paths:
        print(f"No images found in LR directory: {args.lr_dir}"); return

    images_to_process = lr_image_paths
    if args.num_to_plot > 0:
        images_to_process = lr_image_paths[:args.num_to_plot]

    print(f"Attempting to plot {len(images_to_process)} image set(s)...")

    for lr_path in images_to_process:
        lr_filename_full = os.path.basename(lr_path)
        lr_basename_no_ext, lr_ext_with_dot = os.path.splitext(lr_filename_full)
        lr_ext_no_dot = lr_ext_with_dot[1:] if lr_ext_with_dot else ""

        print(f"\nProcessing LR image: {lr_filename_full}")

        lr_img_cv = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        if lr_img_cv is None: print(f"  Could not read LR: {lr_path}"); continue
        
        sr_path = find_sr_image(lr_basename_no_ext, args.sr_dir, args.sr_suffix, args.sr_ext, lr_ext_no_dot)
        sr_img_cv = None; sr_filename_full = "N/A (SR Not Found)"
        if sr_path:
            sr_img_cv = cv2.imread(sr_path, cv2.IMREAD_UNCHANGED)
            if sr_img_cv is not None: sr_filename_full = os.path.basename(sr_path)
            else: print(f"  Found SR path but failed to read: {sr_path}"); sr_filename_full = f"N/A (SR Read Fail: {os.path.basename(sr_path)})"
        else: print(f"  SR image not found for {lr_basename_no_ext} (suffix '{args.sr_suffix}', ext '{args.sr_ext}')")

        hr_img_cv = None; hr_filename_full = "N/A (HR Not Specified or Found)"
        if args.hr_dir:
            hr_path = os.path.join(args.hr_dir, lr_filename_full) # Assumes HR has same name as LR
            if os.path.exists(hr_path):
                hr_img_cv = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
                if hr_img_cv is not None: hr_filename_full = os.path.basename(hr_path)
                else: print(f"  Found HR path but failed to read: {hr_path}"); hr_filename_full = f"N/A (HR Read Fail: {os.path.basename(hr_path)})"
            # else: print(f"  HR image not found at: {hr_path}") # Can be noisy if many HR are missing
        
        display_image_set(lr_img_cv, sr_img_cv, hr_img_cv, 
                           lr_filename_full, sr_filename_full, hr_filename_full)
    print("\nFinished plotting attempts.")

if __name__ == '__main__':
    main()
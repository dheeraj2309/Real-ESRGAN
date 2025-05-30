import argparse
import cv2
import glob
import os
import matplotlib.pyplot as plt

def find_sr_image(lr_img_basename_no_ext, sr_dir, sr_suffix, sr_ext_preference, lr_original_ext_no_dot):
    """
    Attempts to find the corresponding SR image.
    lr_img_basename_no_ext: Basename of the LR image without extension.
    sr_dir: Directory where SR images are stored.
    sr_suffix: Suffix added to SR images (e.g., "_out").
    sr_ext_preference: Preferred extension for SR images ('auto', 'png', 'jpg').
    lr_original_ext_no_dot: Original extension of the LR image without the dot.
    """
    sr_basename = f"{lr_img_basename_no_ext}{sr_suffix}" if sr_suffix else lr_img_basename_no_ext
    
    potential_extensions = []
    if sr_ext_preference == 'auto':
        potential_extensions.extend([lr_original_ext_no_dot, 'png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    else:
        potential_extensions.append(sr_ext_preference)
        if sr_ext_preference != 'png': potential_extensions.append('png') # common fallback

    for ext in potential_extensions:
        sr_path = os.path.join(sr_dir, f"{sr_basename}.{ext}")
        if os.path.exists(sr_path):
            return sr_path
    
    # Fallback: try globbing if simple construction fails (e.g. if RGBA forced png)
    glob_pattern = os.path.join(sr_dir, f"{sr_basename}.*")
    matches = glob.glob(glob_pattern)
    if matches:
        return matches[0] # Return the first match

    return None


def display_image_set(lr_img_cv, sr_img_cv, hr_img_cv, lr_filename, sr_filename, hr_filename):
    """
    Displays LR, SR, and optionally HR images using matplotlib.
    """
    def prep_for_plot(img_cv):
        if img_cv is None:
            return None
        if img_cv.ndim == 2:  # Grayscale
            return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        elif img_cv.shape[2] == 3:  # BGR
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        elif img_cv.shape[2] == 4:  # BGRA
            return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
        return img_cv

    lr_display = prep_for_plot(lr_img_cv)
    sr_display = prep_for_plot(sr_img_cv)
    hr_display = prep_for_plot(hr_img_cv)

    num_images = 0
    if lr_display is not None: num_images +=1
    if sr_display is not None: num_images +=1
    if hr_display is not None: num_images +=1
    
    if num_images == 0:
        print("No images to display for this set.")
        return

    plt.figure(figsize=(7 * num_images, 7))
    plot_idx = 1

    if lr_display is not None:
        plt.subplot(1, num_images, plot_idx)
        plt.imshow(lr_display)
        plt.title(f'LR: {lr_filename}')
        plt.axis('off')
        plot_idx += 1

    if hr_display is not None:
        plt.subplot(1, num_images, plot_idx)
        plt.imshow(hr_display)
        plt.title(f'HR (Ground Truth): {hr_filename}')
        plt.axis('off')
        plot_idx += 1
    
    if sr_display is not None:
        plt.subplot(1, num_images, plot_idx)
        plt.imshow(sr_display)
        plt.title(f'SR (Generated): {sr_filename}')
        plt.axis('off')
        plot_idx += 1
        
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot LR, SR, and optionally HR images.")
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory containing Low-Resolution (LR) images.')
    parser.add_argument('--sr_dir', type=str, required=True, help='Directory containing Super-Resolved (SR) images.')
    parser.add_argument('--hr_dir', type=str, default=None, help='(Optional) Directory containing High-Resolution (HR) ground truth images.')
    parser.add_argument('--num_to_plot', type=int, default=5, help='Number of image sets to plot. 0 for all.')
    parser.add_argument('--sr_suffix', type=str, default='_out', help='Suffix used for SR images (e.g., _out if LR is img.png and SR is img_out.png). Empty string if no suffix.')
    parser.add_argument('--sr_ext', type=str, default='auto', help='Expected extension for SR images (e.g., png, jpg). "auto" tries LR extension then common ones.')

    args = parser.parse_args()

    if not os.path.isdir(args.lr_dir):
        print(f"Error: LR directory not found: {args.lr_dir}")
        return
    if not os.path.isdir(args.sr_dir):
        print(f"Error: SR directory not found: {args.sr_dir}")
        return
    if args.hr_dir and not os.path.isdir(args.hr_dir):
        print(f"Warning: HR directory specified but not found: {args.hr_dir}. HR images will not be plotted.")
        args.hr_dir = None

    lr_image_paths = sorted(glob.glob(os.path.join(args.lr_dir, '*')))
    if not lr_image_paths:
        print(f"No images found in LR directory: {args.lr_dir}")
        return

    count = 0
    for lr_path in lr_image_paths:
        if args.num_to_plot > 0 and count >= args.num_to_plot:
            break

        lr_filename_full = os.path.basename(lr_path)
        lr_basename, lr_ext_with_dot = os.path.splitext(lr_filename_full)
        lr_ext_no_dot = lr_ext_with_dot[1:] if lr_ext_with_dot else ""

        print(f"\nProcessing LR image: {lr_filename_full}")

        # Load LR image
        lr_img_cv = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        if lr_img_cv is None:
            print(f"  Could not read LR image: {lr_path}")
            continue
        
        # Find and load SR image
        sr_path = find_sr_image(lr_basename, args.sr_dir, args.sr_suffix, args.sr_ext, lr_ext_no_dot)
        sr_img_cv = None
        sr_filename_full = "N/A"
        if sr_path:
            sr_img_cv = cv2.imread(sr_path, cv2.IMREAD_UNCHANGED)
            sr_filename_full = os.path.basename(sr_path)
            if sr_img_cv is None:
                print(f"  Found SR image path but could not read: {sr_path}")
            else:
                print(f"  Found SR image: {sr_filename_full}")
        else:
            print(f"  SR image not found for {lr_basename} with suffix '{args.sr_suffix}' in {args.sr_dir}")

        # Find and load HR image (if hr_dir is provided)
        hr_img_cv = None
        hr_filename_full = "N/A"
        if args.hr_dir:
            # HR images usually have the same name and extension as original LR
            hr_path = os.path.join(args.hr_dir, lr_filename_full)
            if os.path.exists(hr_path):
                hr_img_cv = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
                hr_filename_full = os.path.basename(hr_path)
                if hr_img_cv is None:
                    print(f"  Found HR image path but could not read: {hr_path}")
                else:
                    print(f"  Found HR image: {hr_filename_full}")
            else:
                print(f"  HR image not found at: {hr_path}")
        
        display_image_set(lr_img_cv, sr_img_cv, hr_img_cv, 
                          lr_filename_full, sr_filename_full, hr_filename_full)
        count += 1

    if count == 0:
        print("No images were processed or plotted.")
    else:
        print(f"\nPlotted {count} image set(s).")

if __name__ == '__main__':
    main()
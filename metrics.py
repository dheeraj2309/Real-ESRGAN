import os
import re
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def get_image_files(folder_path):
    """Returns a sorted list of image file paths in a folder."""
    files = []
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder not found or is not a directory: {folder_path}")
        return files
    for f_name in sorted(os.listdir(folder_path)): # Sort to help with ordered matching
        if f_name.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(folder_path, f_name))
    return files

def find_image_pairs(sr_folder, gt_folder, matching_strategy="strict_name"):
    """
    Finds corresponding pairs of images between SR and GT folders.

    Args:
        sr_folder (str): Path to the super-resolved images folder.
        gt_folder (str): Path to the ground-truth images folder.
        matching_strategy (str):
            "strict_name": SR and GT images must have the exact same filename.
            "sorted_order": Pairs images based on their sorted order in the folders.
                           Useful if names differ but correspondence is by sequence.
            "common_prefix_numerical": Tries to match based on a common prefix followed by numbers.
                                       e.g., 'img_001_sr.png' and 'img_001_gt.png'
                                       (This is a more advanced example, let's start simpler)

    Returns:
        list: A list of tuples, where each tuple is (sr_image_path, gt_image_path).
    """
    sr_files = get_image_files(sr_folder)
    gt_files = get_image_files(gt_folder)
    pairs = []

    if not sr_files:
        print(f"No image files found in SR folder: {sr_folder}")
        return pairs
    if not gt_files:
        print(f"No image files found in GT folder: {gt_folder}")
        return pairs

    print(f"\nAttempting to match images using strategy: '{matching_strategy}'")

    if matching_strategy == "strict_name":
        gt_filenames_map = {os.path.basename(f): f for f in gt_files}
        for sr_path in sr_files:
            sr_filename = os.path.basename(sr_path)
            if sr_filename in gt_filenames_map:
                pairs.append((sr_path, gt_filenames_map[sr_filename]))
            else:
                print(f"Warning: No matching GT image for SR image (strict_name): {sr_filename}")
        if not pairs:
             print("No pairs found with 'strict_name'. Try 'sorted_order' if applicable.")


    elif matching_strategy == "sorted_order":
        if len(sr_files) != len(gt_files):
            print(f"Warning: Number of SR images ({len(sr_files)}) "
                  f"does not match number of GT images ({len(gt_files)}). "
                  "Pairing based on the minimum count.")
        
        min_len = min(len(sr_files), len(gt_files))
        for i in range(min_len):
            pairs.append((sr_files[i], gt_files[i]))
            # Optional: print what's being paired for verification
            # print(f"  Pairing by order: {os.path.basename(sr_files[i])} <-> {os.path.basename(gt_files[i])}")
        if not pairs and (sr_files and gt_files):
            print("Could not form any pairs with 'sorted_order'. Check folder contents and sorting.")
        elif min_len == 0 and (sr_files or gt_files):
             print("One of the folders is empty or contains no images, no pairs formed.")


    # Example of a more complex strategy (can be expanded)
    # elif matching_strategy == "common_alphanum_stem":
    #     # Extracts alphanumeric base name, e.g., "image123" from "image_123_sr.png"
    #     def get_stem(filename):
    #         name_part = os.path.splitext(os.path.basename(filename))[0]
    #         # Remove common SR/GT suffixes or prefixes if they exist
    #         name_part = re.sub(r'_sr$|_gt$|^sr_|^gt_', '', name_part, flags=re.IGNORECASE)
    #         # Keep only alphanumeric characters for a cleaner stem
    #         stem = re.sub(r'[^a-zA-Z0-9]', '', name_part)
    #         return stem.lower() # Case-insensitive matching

    #     gt_stems_map = {get_stem(f): f for f in gt_files}
    #     for sr_path in sr_files:
    #         sr_stem = get_stem(sr_path)
    #         if sr_stem and sr_stem in gt_stems_map:
    #             pairs.append((sr_path, gt_stems_map[sr_stem]))
    #             # print(f"  Paired by stem '{sr_stem}': {os.path.basename(sr_path)} <-> {os.path.basename(gt_stems_map[sr_stem])}")
    #         else:
    #             print(f"Warning: No GT match for SR stem '{sr_stem}' from file: {os.path.basename(sr_path)}")


    else:
        print(f"Error: Unknown matching strategy '{matching_strategy}'")

    if pairs:
        print(f"Found {len(pairs)} image pairs.")
    else:
        print("No image pairs were found with the selected strategy.")
    return pairs


def calculate_metrics_for_pairs(image_pairs):
    """
    Calculates PSNR and SSIM for the given image pairs.
    """
    psnr_values = []
    ssim_values = []
    processed_files = 0

    if not image_pairs:
        print("No image pairs to process.")
        return None, None

    print("\n--- Calculating Metrics ---")
    for sr_image_path, gt_image_path in image_pairs:
        try:
            img_sr_pil = Image.open(sr_image_path)
            img_gt_pil = Image.open(gt_image_path)

            img_sr = np.array(img_sr_pil)
            img_gt = np.array(img_gt_pil)

            if img_sr.shape != img_gt.shape:
                print(f"Warning: Dimensions mismatch for pair: "
                      f"{os.path.basename(sr_image_path)} (SR: {img_sr.shape}) and "
                      f"{os.path.basename(gt_image_path)} (GT: {img_gt.shape}). Skipping.")
                # Optional: Attempt to resize sr_image to gt_image dimensions
                # from skimage.transform import resize
                # print(f"Attempting to resize SR image to GT dimensions: {img_gt.shape}")
                # img_sr = resize(img_sr, img_gt.shape, anti_aliasing=True, preserve_range=True)
                # if img_sr.dtype != img_gt.dtype: # Ensure dtype matches after resize
                #     img_sr = img_sr.astype(img_gt.dtype)
                # if img_sr.shape != img_gt.shape: # Check again
                #     print("Resize failed or dimensions still mismatch. Skipping.")
                #     continue
                continue


            # Determine data range
            data_range = 255.0 if img_gt.max() > 1.0 else 1.0
            # If images are already float 0-1, skimage handles it well.
            # If they are uint8 (0-255), data_range=255.0 is correct.

            current_psnr = psnr(img_gt, img_sr, data_range=data_range)
            psnr_values.append(current_psnr)

            # For SSIM, scikit-image prefers float images in [0,1] or [-1,1] for some versions/defaults,
            # but works with integer types if data_range is correctly set.
            # win_size must be odd and smaller than image dimensions. Default is 7.
            # Ensure win_size is not too large for small images
            min_dim = min(img_gt.shape[0], img_gt.shape[1])
            win_s = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1) # Ensure win_size is odd and <= min_dim
            if win_s < 3 : # SSIM window must be at least 3x3 for some modes
                print(f"Warning: Image dimensions too small for default SSIM window ({os.path.basename(gt_image_path)}: {img_gt.shape}). Skipping SSIM.")
                current_ssim = float('nan') # Or handle as you see fit
            else:
                if img_gt.ndim == 3: # Color image
                    current_ssim = ssim(img_gt, img_sr, data_range=data_range, channel_axis=-1, win_size=win_s, gaussian_weights=True)
                elif img_gt.ndim == 2: # Grayscale image
                    current_ssim = ssim(img_gt, img_sr, data_range=data_range, win_size=win_s, gaussian_weights=True)
                else:
                    print(f"Warning: Unsupported image dimensions for SSIM ({img_gt.ndim}) for {os.path.basename(gt_image_path)}. Skipping SSIM.")
                    current_ssim = float('nan')
            
            if not np.isnan(current_ssim):
                ssim_values.append(current_ssim)

            print(f"SR: {os.path.basename(sr_image_path)}, GT: {os.path.basename(gt_image_path)} "
                  f"- PSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}")
            processed_files += 1

        except FileNotFoundError:
            print(f"Error: File not found during metric calculation. SR: {sr_image_path}, GT: {gt_image_path}")
        except Exception as e:
            print(f"Error processing pair ({os.path.basename(sr_image_path)}, {os.path.basename(gt_image_path)}): {e}")

    if processed_files > 0:
        # Filter out NaNs from ssim_values if any occurred
        valid_ssim_values = [s for s in ssim_values if not np.isnan(s)]
        
        avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')
        avg_ssim = np.mean(valid_ssim_values) if valid_ssim_values else float('nan')
        
        print(f"\n--- Average Metrics ({processed_files} successfully processed pairs) ---")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        return avg_psnr, avg_ssim
    else:
        print("\nNo images were successfully processed for metrics.")
        return None, None

if __name__ == "__main__":
    # --- Configuration ---
    SUPER_RESOLVED_FOLDER = "downloaded_images"
    GROUND_TRUTH_FOLDER = "test"

    # Choose your matching strategy:
    # "strict_name": Filenames must be identical.
    # "sorted_order": Files are paired based on sorted order in their folders.
    #                 (e.g., sr_001.png with gt_001.png, or first SR with first GT)
    # "common_alphanum_stem": (More advanced, currently commented out but can be enabled/refined)
    #                      Tries to match based on a common alphanumeric part of the name.
    MATCHING_STRATEGY = "strict_name" # Or "sorted_order"
    # --- End Configuration ---

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sr_folder_path = os.path.join(script_dir, SUPER_RESOLVED_FOLDER)
    gt_folder_path = os.path.join(script_dir, GROUND_TRUTH_FOLDER)

    # 1. Find image pairs
    image_pairs_to_process = find_image_pairs(sr_folder_path, gt_folder_path, MATCHING_STRATEGY)

    # 2. Calculate metrics for these pairs
    if image_pairs_to_process:
        calculate_metrics_for_pairs(image_pairs_to_process)
    else:
        print(f"No image pairs found using strategy '{MATCHING_STRATEGY}'. Cannot calculate metrics.")
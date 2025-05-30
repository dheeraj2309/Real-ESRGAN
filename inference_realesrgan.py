# --- START OF MODIFIED inference_realesrgan.py ---
import argparse
import cv2
import glob
import os
import numpy as np # Added
from skimage.metrics import peak_signal_noise_ratio, structural_similarity # Added
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    """Inference demo for Real-ESRGAN with PSNR/SSIM evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder (LR images)')
    parser.add_argument(
        '--hr_folder', type=str, default=None, help='Folder containing High-Resolution ground truth images for PSNR/SSIM calculation') # Added
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder for SR images')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image (e.g., _out -> imgname_out.png)')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension for saving SR images. Options: auto | jpg | png. "auto" means using the same extension as input LR, or png for RGBA.')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f'Model {args.model_name} not supported.')

    # Determine model paths
    # model_path_final will store the path to be used by the upsampler
    if args.model_path is not None:
        model_path_final = args.model_path
        if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            # If user provides a model_path for realesr-general-x4v3, they must ensure
            # the wdn model is also available relative to this path, or this replace might fail.
             wdn_model_path = model_path_final.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
             if not os.path.isfile(wdn_model_path):
                 print(f"Warning: WDN model for DNI not found at {wdn_model_path}. Denoise strength might not work as expected.")
    else:
        # Default path in 'weights' directory
        # For 'realesr-general-x4v3', model_path_final will point to the main model after downloads
        # The wdn model is also downloaded by the loop if file_url contains it.
        # The name of the last file in file_url is what model_path_final gets if multiple downloads happen for one model_name.
        
        # Construct the primary model filename
        primary_model_filename = args.model_name + '.pth'
        model_path_final = os.path.join('weights', primary_model_filename)

        # Check if all necessary files (could be multiple for some models) exist
        # For 'realesr-general-x4v3', this means checking for both .pth files if dni is used.
        all_files_exist = True
        # For simplicity, we check the primary model path first.
        # If DNI is used, the wdn model path will be constructed later.
        if not os.path.isfile(model_path_final):
             all_files_exist = False # At least the primary is missing
        
        if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            wdn_model_filename = args.model_name.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3') + '.pth'
            if not os.path.isfile(os.path.join('weights', wdn_model_filename)):
                all_files_exist = False


        if not all_files_exist:
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            weights_dir = os.path.join(ROOT_DIR, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            downloaded_paths = []
            for url in file_url:
                # load_file_from_url will use the filename from the URL
                downloaded_path = load_file_from_url(
                    url=url, model_dir=weights_dir, progress=True, file_name=None)
                downloaded_paths.append(downloaded_path)
            
            # After downloads, model_path_final should point to the primary model for the specified args.model_name
            # This usually means the one matching args.model_name + '.pth'
            # If file_url had multiple files, the last one downloaded is what 'downloaded_path' would be.
            # Let's explicitly set model_path_final to the expected primary model name.
            model_path_final = os.path.join(weights_dir, primary_model_filename)
            if not os.path.isfile(model_path_final) and downloaded_paths:
                # Fallback if primary model name logic fails (e.g. URL name different)
                # For single file_url, this is direct. For multiple, it depends on which one is "primary".
                # The RealESRGAN_x4plus etc. have one URL.
                # 'realesr-general-x4v3' has two, its primary is 'realesr-general-x4v3.pth'.
                # The file_url is ordered such that the main model is last for 'realesr-general-x4v3'.
                if primary_model_filename == os.path.basename(downloaded_paths[-1]):
                     model_path_final = downloaded_paths[-1]
                else: # Try to find it
                    for dp in downloaded_paths:
                        if primary_model_filename == os.path.basename(dp):
                            model_path_final = dp
                            break
                    else: # If still not found by exact name match (should not happen if URLs are correct)
                        print(f"Warning: Primary model {primary_model_filename} not found after download. Using last downloaded: {downloaded_paths[-1] if downloaded_paths else 'None'}")
                        if downloaded_paths: model_path_final = downloaded_paths[-1]


    # use dni to control the denoise strength
    dni_weight = None
    model_path_for_upsampler = model_path_final # This will be passed to RealESRGANer

    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        # model_path_final should be the path to 'realesr-general-x4v3.pth'
        wdn_model_path = model_path_final.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        # Ensure wdn_model_path actually exists (it should if downloaded correctly or provided)
        if not os.path.isfile(wdn_model_path):
            print(f"Warning: DNI weight specified, but WDN model not found at derived path: {wdn_model_path}")
            print("Denoising might not use the WDN model. Ensure it's available.")
            # Proceed without DNI or let RealESRGANer handle missing file if it checks.
            # For safety, let RealESRGANer get only the main model path if WDN is missing.
        else:
            model_path_for_upsampler = [model_path_final, wdn_model_path]
            dni_weight = [args.denoise_strength, 1 - args.denoise_strength]


    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path_for_upsampler,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        except ImportError:
            print("INFO: GFPGAN not found. Face enhancement disabled. To install: pip install gfpgan")
            args.face_enhance = False
            
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    
    psnr_values = []
    ssim_values = []

    for idx, path in enumerate(paths):
        imgname, original_lr_extension_with_dot = os.path.splitext(os.path.basename(path))
        print(f'Processing {idx + 1}/{len(paths)}: {imgname}{original_lr_extension_with_dot}')

        img_lr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img_lr is None:
            print(f"Warning: Failed to read LR image {path}. Skipping.")
            continue

        img_mode_rgba = len(img_lr.shape) == 3 and img_lr.shape[2] == 4

        img_sr = None
        try:
            if args.face_enhance:
                _, _, img_sr = face_enhancer.enhance(img_lr, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                img_sr, _ = upsampler.enhance(img_lr, outscale=args.outscale)
        except RuntimeError as error:
            print('Error during SR:', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            continue 
        except Exception as error:
            print(f'An error occurred during SR processing {imgname}: {error}')
            continue
        
        if img_sr is None:
            print(f"Warning: SR image generation failed for {imgname}. Skipping.")
            continue

        if args.ext == 'auto':
            current_output_extension = original_lr_extension_with_dot[1:] if original_lr_extension_with_dot else 'png'
        else:
            current_output_extension = args.ext
        
        if img_mode_rgba or (len(img_sr.shape) == 3 and img_sr.shape[2] == 4):
            current_output_extension = 'png'
        
        sr_basename = f"{imgname}_{args.suffix}" if args.suffix else imgname
        save_path = os.path.join(args.output, f'{sr_basename}.{current_output_extension}')
        
        cv2.imwrite(save_path, img_sr)
        print(f"Saved SR image to: {save_path}")

        if args.hr_folder:
            hr_img_path_found = None
            possible_hr_extensions = [original_lr_extension_with_dot, '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            # Prioritize exact extension match if original_lr_extension_with_dot is not empty
            if original_lr_extension_with_dot:
                 # Ensure it's not already in the list to avoid double check
                if original_lr_extension_with_dot not in possible_hr_extensions:
                    possible_hr_extensions.insert(0, original_lr_extension_with_dot)
                elif possible_hr_extensions.index(original_lr_extension_with_dot) != 0:
                    possible_hr_extensions.pop(possible_hr_extensions.index(original_lr_extension_with_dot))
                    possible_hr_extensions.insert(0, original_lr_extension_with_dot)


            for ext_try in possible_hr_extensions:
                hr_path_candidate = os.path.join(args.hr_folder, f"{imgname}{ext_try}")
                if os.path.isfile(hr_path_candidate):
                    hr_img_path_found = hr_path_candidate
                    break
            
            if not hr_img_path_found:
                print(f"Warning: HR image for {imgname} not found in {args.hr_folder}. Skipping metrics.")
                continue

            img_hr = cv2.imread(hr_img_path_found, cv2.IMREAD_UNCHANGED)
            if img_hr is None:
                print(f"Warning: Failed to read HR image {hr_img_path_found}. Skipping metrics.")
                continue

            img_sr_metric = img_sr[:, :, :3] if len(img_sr.shape) == 3 and img_sr.shape[2] == 4 else img_sr
            img_hr_metric = img_hr[:, :, :3] if len(img_hr.shape) == 3 and img_hr.shape[2] == 4 else img_hr

            if img_sr_metric.shape != img_hr_metric.shape:
                print(f"Warning: SR shape {img_sr_metric.shape} and HR shape {img_hr_metric.shape} mismatch for {imgname}. Resizing HR to SR size for metrics.")
                img_hr_metric = cv2.resize(img_hr_metric, (img_sr_metric.shape[1], img_sr_metric.shape[0]), interpolation=cv2.INTER_AREA)

            if len(img_sr_metric.shape) == 2: img_sr_metric = cv2.cvtColor(img_sr_metric, cv2.COLOR_GRAY2BGR)
            if len(img_hr_metric.shape) == 2: img_hr_metric = cv2.cvtColor(img_hr_metric, cv2.COLOR_GRAY2BGR)
            
            if img_sr_metric.ndim == 2: # If still 2D (e.g. user provides grayscale and alpha), convert to 3-channel
                 img_sr_metric = cv2.cvtColor(img_sr_metric, cv2.COLOR_GRAY2BGR)
            if img_hr_metric.ndim == 2:
                 img_hr_metric = cv2.cvtColor(img_hr_metric, cv2.COLOR_GRAY2BGR)


            if img_sr_metric.shape[2] != img_hr_metric.shape[2] or img_sr_metric.shape[2] !=3 : # Ensure both are 3-channel BGR
                 print(f"Warning: Channel mismatch or not 3 channels for metric calculation for {imgname}. SR channels: {img_sr_metric.shape}, HR channels: {img_hr_metric.shape}. Skipping metrics.")
                 continue
            
            try:
                psnr = peak_signal_noise_ratio(img_hr_metric, img_sr_metric, data_range=255)
                psnr_values.append(psnr)

                win_size = min(7, min(img_sr_metric.shape[0], img_sr_metric.shape[1]))
                if win_size % 2 == 0: win_size -=1
                
                ssim = np.nan
                if win_size >= 3:
                    import skimage
                    skimage_version = list(map(int, skimage.__version__.split('.')))
                    if skimage_version[0] == 0 and skimage_version[1] < 19:
                         ssim = structural_similarity(img_hr_metric, img_sr_metric, data_range=255, multichannel=True, win_size=win_size)
                    else:
                         ssim = structural_similarity(img_hr_metric, img_sr_metric, data_range=255, channel_axis=-1, win_size=win_size)
                else:
                    print(f"Warning: Image {imgname} too small (min_dim < 3 after processing) for SSIM. Skipping SSIM.")
                
                ssim_values.append(ssim) # ssim can be np.nan here
                print(f"Metrics for {imgname}: PSNR = {psnr:.4f} dB, SSIM = {ssim:.4f if not np.isnan(ssim) else 'N/A'}")

            except Exception as e:
                print(f"Error calculating metrics for {imgname}: {e}")

    if args.hr_folder:
        if psnr_values:
            avg_psnr = np.nanmean([p for p in psnr_values if not np.isnan(p)])
            print(f"\nAverage PSNR: {avg_psnr:.4f} dB (over {len([p for p in psnr_values if not np.isnan(p)])} images)")
        else:
            print("\nNo PSNR values were calculated.")
        
        if ssim_values:
            avg_ssim = np.nanmean([s for s in ssim_values if not np.isnan(s)])
            print(f"Average SSIM: {avg_ssim:.4f} (over {len([s for s in ssim_values if not np.isnan(s)])} images)")
        else:
            print("No SSIM values were calculated.")

    print(f"\nAll images processed. SR results saved in '{args.output}'.")
    if args.hr_folder:
        print("PSNR/SSIM metrics calculated against HR images from '{}'.".format(args.hr_folder))

if __name__ == '__main__':
    main()
# --- END OF MODIFIED inference_realesrgan.py ---
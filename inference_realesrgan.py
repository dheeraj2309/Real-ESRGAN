# --- START OF FILE inference_realesrgan.py ---

import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Import for plotting
import matplotlib.pyplot as plt

def display_images(lr_img, sr_img, lr_img_basename, sr_img_basename, hr_img=None, hr_img_basename=None):
    """
    Displays LR, SR, and optionally HR images using matplotlib.
    lr_img: OpenCV image (numpy array) for Low Resolution.
    sr_img: OpenCV image (numpy array) for Super Resolved.
    lr_img_basename: Filename for LR image title.
    sr_img_basename: Filename for SR image title.
    hr_img: OpenCV image (numpy array) for High Resolution (Ground Truth), optional.
    hr_img_basename: Filename for HR image title, optional.
    """
    # Helper to convert OpenCV BGR/BGRA/Gray to Matplotlib RGB/RGBA
    def prep_for_plot(img_cv):
        if img_cv is None:
            return None
        if img_cv.ndim == 2:  # Grayscale
            return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        elif img_cv.shape[2] == 3:  # BGR
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        elif img_cv.shape[2] == 4:  # BGRA
            return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
        return img_cv  # Should not happen with typical image formats

    lr_display = prep_for_plot(lr_img)
    sr_display = prep_for_plot(sr_img)
    hr_display = prep_for_plot(hr_img) if hr_img is not None else None

    num_images = 2
    if hr_display is not None:
        num_images = 3

    plt.figure(figsize=(7 * num_images, 7)) # Adjust figure size based on number of images

    # Plot LR image
    plt.subplot(1, num_images, 1)
    plt.imshow(lr_display)
    plt.title(f'Original LR: {lr_img_basename}')
    plt.axis('off')

    if hr_display is not None:
        # Plot HR image
        plt.subplot(1, num_images, 2)
        plt.imshow(hr_display)
        plt.title(f'Ground Truth HR: {hr_img_basename}')
        plt.axis('off')
        # Plot SR image (as the third image)
        plt.subplot(1, num_images, 3)
        plt.imshow(sr_display)
        plt.title(f'Super-Resolved: {sr_img_basename}')
        plt.axis('off')
    else:
        # Plot SR image (as the second image)
        plt.subplot(1, num_images, 2)
        plt.imshow(sr_display)
        plt.title(f'Super-Resolved: {sr_img_basename}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder (LR images)')
    parser.add_argument(
        '--hr_input',
        type=str,
        default=None,
        help='Path to the Ground Truth High-Resolution (HR) image folder. Used if --plot_images is active.')
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
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
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
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument(
        '--plot_images', action='store_true', help='Plot the original LR, Ground Truth HR (if available), and Super-Resolved images.')

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


    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            # check if 'weights' directory exists, if not create it
            os.makedirs(os.path.join(ROOT_DIR, 'weights'), exist_ok=True)
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, original_lr_extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # This is the LR image
        
        if img is None:
            print(f"Warning: Failed to read LR image {path}. Skipping.")
            continue

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale) # This is the SR image
        except RuntimeError as error:
            print('Error during SR:', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            continue # Skip saving and plotting for this image
        except Exception as error:
            print(f'An error occurred during SR processing {imgname}: {error}')
            continue # Skip saving and plotting for this image
        else:
            # Determine output extension
            if args.ext == 'auto':
                current_output_extension = original_lr_extension[1:] # remove dot
            else:
                current_output_extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                current_output_extension = 'png'
            
            # Construct save path for SR image
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{current_output_extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{current_output_extension}')
            
            cv2.imwrite(save_path, output)

            # Plotting logic
            if args.plot_images:
                hr_img = None
                hr_img_basename = None
                if args.hr_input:
                    # Construct path to the corresponding HR image
                    # Assumes HR image has the same base name and original extension as LR image
                    hr_image_path = os.path.join(args.hr_input, f'{imgname}{original_lr_extension}')
                    if os.path.exists(hr_image_path):
                        hr_img = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED)
                        if hr_img is None:
                            print(f"Warning: Failed to read HR image {hr_image_path}, though file exists.")
                        else:
                             hr_img_basename = os.path.basename(hr_image_path)
                    else:
                        print(f"Warning: Ground Truth HR image not found at {hr_image_path}")
                
                try:
                    display_images(
                        lr_img=img, 
                        sr_img=output, 
                        lr_img_basename=os.path.basename(path),
                        sr_img_basename=os.path.basename(save_path),
                        hr_img=hr_img,
                        hr_img_basename=hr_img_basename
                    )
                except Exception as e:
                    print(f"Error during plotting for {imgname}: {e}")


if __name__ == '__main__':
    main()
# --- END OF FILE inference_realesrgan.py ---
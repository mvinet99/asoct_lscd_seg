# Perform prepcrocessing steps for the ASOCT pipeline
import os
import skimage
import cv2 as cv
from tqdm import tqdm
from skimage import exposure, img_as_float, img_as_ubyte

def ensure_folder_exists(parent_dir, folder_names):
    """
    Create needed folders for the project pipeline
    """
    for folder_name in folder_names:
        folder_path = os.path.join(parent_dir, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            continue

def list_unet_folders(num_models):
    """
    Create folders for training MSU_Net model
    """
    subfolders = [
        "predicted_masks",
    ]

    all_paths = []

    for i in range(1, num_models + 1):
        base = f"UNet/UNet{i}"
        for sub in subfolders:
            path = f"{base}/{sub}"
            all_paths.append(path)

    return all_paths

def resize_im(data_path):
    """
    Resize OCT images to correct size for the project
    """
    files = os.listdir(data_path + '/raw_corneal/')
    resized_path = os.path.join(data_path, 'resized')

    for file in tqdm(files):
        output_file = os.path.join(resized_path, file)

        if os.path.exists(output_file):
            continue
        
        im = cv.imread(data_path+'raw_corneal/' + file, cv.IMREAD_GRAYSCALE)
        resized_im = cv.resize(im, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)
        cv.imwrite(output_file, resized_im)

def normalize(data_path):
    """
    Normalize OCT images using histogram equalization
    """
    files = sorted(os.listdir(data_path+'/smoothed/')) 

    og_path = data_path+'/smoothed/'
    normal_path = data_path + '/normal/'

    # Load and prepare reference image
    img_path = '/radraid2/mvinet/ASOCT/Current/data/crossline_02_02_26/smoothed/S0001894_slice_1.png'
    ref_raw = cv.imread(img_path)
    ref_raw = cv.resize(ref_raw, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)

    # Perform equalization
    ref_img = exposure.equalize_adapthist(ref_raw, clip_limit=0.005, kernel_size=[820,1], nbins=256) 

    for file in tqdm(files):
        input_file = og_path + file
        output_file = normal_path + file

        image = cv.imread(input_file)
        image = cv.resize(image, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)

        # Convert the current image to float to match the ref_img dtype
        image_float = img_as_float(image)
        
        matched_image = exposure.match_histograms(image_float, ref_img)

        # Convert back to uint8 [0, 255] for OpenCV saving
        matched_image_uint8 = img_as_ubyte(matched_image)

        cv.imwrite(output_file, matched_image_uint8)

def preprocess(data_path, result_path, folds):
    """
    Preprocessing: ensure necesssary folders exist, create images needed for mask creation script
    """
    
    print('Running preprocessing')

    # Create needed paths for the pipeline - base data files
    ensure_folder_exists(data_path, ['edge', 'normal', 'raw', 'resized'])
    
    #Create X number of UNet folders where X is the number of folds
    unet_paths = list_unet_folders(folds)

    # Create needed paths for the pipeline - result files
    ensure_folder_exists(result_path, ['all_masks_overlays', 'mask_slices', 'reconstructed_masks', 'reconstructed_predictions',
                                     'thickness_npy', 'thickness_outlines', 'reconstructed_npy', 'patch_thickness',
                                     'eval_images', 'blank_mask_outlines', 'mask_overlays_outlines', 'reconstructed_predictions_multiclass_color',
                                     'patches', 'patches/images/', 'patches/masks/', 'thickness_outlines_num']
                                     + unet_paths)

    # Upscale and resize images to correct size
    resize_im(data_path)

    # Create normalized images from resized images
    normalize(data_path)

import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy


def get_RGB_position(raw_pattern):
    """ 
    input: raw_pattern (np.ndarray) - The raw pattern
    output: R, Gr_pos, B, Gb_pos - Green channel positions
    """

    assert(raw_pattern.shape == (2,2))

    # Positions in the numpy file
    R, Gr, B, Gb  = 0, 1, 2, 3
    
    # Get the positinos from the raw_pattern
    R_pos = np.squeeze(np.argwhere(raw_pattern == R))
    Gr_pos = np.squeeze(np.argwhere(raw_pattern == Gr))
    B_pos = np.squeeze(np.argwhere(raw_pattern == B))
    Gb_pos = np.squeeze(np.argwhere(raw_pattern == Gb))

    return R_pos, Gr_pos, B_pos, Gb_pos

def rgb_to_bgr(img):
    bgr_img = np.zeros_like(img)
    bgr_img[:, :, 0] = img[:, :, 2]
    bgr_img[:, :, 1] = img[:, :, 1]
    bgr_img[:, :, 2] = img[:, :, 0]

    return bgr_img

def bgr_to_rgb(img):
    return rgb_to_bgr(img)

def get_raw(input_dir, use_temp, verbose=True):
    """
    Input: (str) input_dir
    Output: (np.ndarray) raw_imgs

    Finds all the dng files in the given directory and converts them to a stack of raw images after using dcraw internally.

    Note: This functions needs dcraw to be installed in your system.
    """
    if verbose:
        print("Loading Raw images...")
    

    if (use_temp == 1):
        raw_exists = os.path.isfile("./temp/raw_imgs.npy")
        if raw_exists:
            raw_imgs = np.load("./temp/raw_imgs.npy")
        else:
            use_temp = 0

    if (use_temp == 0):
        # Find all the .dng files
        dng_files = glob.glob(input_dir + "/*.dng")
        dng_files.sort()

        # Check if it is older than 2016, if yes remove the last file as it is an overexposed image
        year_list = ["2014", "2015"]
        is_high_exposure = any(yr in input_dir for yr in year_list)
        
        if is_high_exposure:
            del dng_files[-1]

        # Run dcraw command to get tiffs
        dcraw_command = "dcraw -4 -D -T {}"
        for dng_dir in tqdm(dng_files):
            os.system(dcraw_command.format(dng_dir))
    
        # Find all the .tiff files
        raw_files = glob.glob(input_dir + "/payload*.tiff")
        raw_files.sort()

        # Read all raw files using plt
        raw_imgs = []
        for raw_dir in raw_files:
            raw = plt.imread(raw_dir)
            raw_imgs.append(raw)

        raw_imgs = np.stack(raw_imgs)
        np.save("./temp/raw_imgs.npy", raw_imgs)

    return raw_imgs

def get_raw_object(input_dir, ref_id=0):
    """ 
    Input: input_dir (str) - Path to where the input files are present
    Output: rawpy_obj
    
    It returns the rawpy object which can be used to determine color matrix, white balance and other such parameter as needed
    """
    raw_files = glob.glob(input_dir + "/*.dng")
    raw_files.sort()

    # Use reference frame as the object
    rp_im = rawpy.imread(raw_files[ref_id])
    return rp_im

def mosaic_image(color_planes, raw_pattern):
    R, Gr, B, Gb = get_RGB_position(raw_pattern)
    C, H, W = color_planes.shape

    assert(C == 4)

    mosaic_frame = np.zeros((H*2, W *2))

    mosaic_frame[R[0]::2, R[1]::2] = color_planes[0, ...]
    mosaic_frame[Gr[0]::2, Gr[1]::2] = color_planes[1, ...]
    mosaic_frame[B[0]::2, B[1]::2] = color_planes[2, ...]
    mosaic_frame[Gb[0]::2, Gb[0]::2] = color_planes[3, ...]

    return mosaic_frame

def demosaic_image(raw, raw_pattern):
    R, Gr, B, Gb = get_RGB_position(raw_pattern)

    R_frame = raw[R[0]::2, R[1]::2]
    Gr_frame = raw[Gr[0]::2, Gr[1]::2]
    B_frame = raw[B[0]::2, B[1]::2]
    Gb_frame = raw[Gb[0]::2, Gb[0]::2]

    R_full_frame = cv2.resize(R_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    Gr_full_frame = cv2.resize(Gr_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    B_full_frame = cv2.resize(B_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    Gb_full_frame = cv2.resize(Gb_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    G_full_frame = (Gr_full_frame + Gb_full_frame) / 2

    return np.stack((R_full_frame, G_full_frame, B_full_frame), axis=2)

def apply_white_balance(raw, raw_obj):
    # Get the multipliers from the camera
    wb_multipliers = raw_obj.camera_whitebalance

    R, Gr, B, Gb = get_RGB_position(raw_obj.raw_pattern)

    # TODO: Q: Why is the last green plane multiplier 0?
    wb_multipliers[-1] = 1.0

    # Multiply the channels with the multipliers
    raw[R[0]::2, R[1]::2] = raw[R[0]::2, R[1]::2] * wb_multipliers[0]
    raw[Gr[0]::2, Gr[1]::2] = raw[Gr[0]::2, Gr[1]::2] * wb_multipliers[1]
    raw[B[0]::2, B[1]::2] = raw[B[0]::2, B[1]::2] * wb_multipliers[2]
    raw[Gb[0]::2, Gb[0]::2] = raw[Gb[0]::2, Gb[0]::2] * wb_multipliers[3]

    # Normalize the channels
    raw[R[0]::2, R[1]::2] = raw[R[0]::2, R[1]::2] / np.max(raw[R[0]::2, R[1]::2])
    raw[Gr[0]::2, Gr[1]::2] = raw[Gr[0]::2, Gr[1]::2] / np.max(raw[Gr[0]::2, Gr[1]::2])
    raw[B[0]::2, B[1]::2] = raw[B[0]::2, B[1]::2] / np.max(raw[B[0]::2, B[1]::2])
    raw[Gb[0]::2, Gb[0]::2] = raw[Gb[0]::2, Gb[0]::2] / np.max(raw[Gb[0]::2, Gb[0]::2])

    return raw

def lens_shading_correction(img, ref_id):
    pass

def chroma_denoising(img):
    # TODO: Loss of precision here, is there somthing better that can be done?
    yuv_img = cv2.cvtColor(img.astype(np.single), cv2.COLOR_RGB2YUV)

    # TODO: Check strength of bilateral filter
    yuv_img[:, :, 1] = cv2.bilateralFilter(yuv_img[:, :, 1], d=3, sigmaColor=30, sigmaSpace=30)
    yuv_img[:, :, 2] = cv2.bilateralFilter(yuv_img[:, :, 2], d=3, sigmaColor=30, sigmaSpace=30)

    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)

    return rgb_img
    
def color_correction(img, in_path):
    rgb2rgb = np.loadtxt("{}/rgb2rgb.txt".format(in_path))
    rgb2rgb = rgb2rgb.reshape(3,3)

    # TODO: Is this right?
    cc_img = rgb2rgb @ img.transpose([2,1,0]).reshape(3, -1)

    cc_img = cc_img.reshape(3, img.shape[1], img.shape[0]).transpose([2,1,0])

    # TODO: Should we normalize?
    cc_img = cc_img / np.max(cc_img, axis=(0,1)).reshape(1,-1)

    return cc_img

def process(merged_raw, ref_id, raw_obj, args):
    # White balance the raw
    wb_raw = apply_white_balance(merged_raw, raw_obj)

    # Demosaic image
    dmsc_img = demosaic_image(wb_raw, raw_obj.raw_pattern)

    # Chroma denoising
    rgb_img = chroma_denoising(dmsc_img)

    # Perform Color Correction
    cc_img = color_correction(rgb_img, args.input)
    
    

    return cc_img






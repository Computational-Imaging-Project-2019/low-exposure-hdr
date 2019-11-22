import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy


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






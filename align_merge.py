import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy

def get_GG_position(raw_pattern):
    """ input: raw_pattern (np.ndarray) - The raw pattern
    output: Gr_pos, Gb_pos - Green channel positions"""
    
    assert(raw_pattern.shape == (2,2))

    Gr, Gb  = 1, 3

    Gr_pos = np.squeeze(np.argwhere(raw_pattern == Gr))
    Gb_pos = np.squeeze(np.argwhere(raw_pattern == Gb))
    return Gr_pos, Gb_pos

def select_ref_frame(raw_imgs, raw_pattern):
    """ Input: raw_imgs (np.ndarray) - Raw images list
        Output: index (int) - integer index to the raw images list
        
        This function returns the selected reference frame index 
        after selecting it using a laplacian measure of  sharpness"""

    assert(raw_pattern.shape == (2,2))
    assert(raw_imgs.shape[0] >= 3)

    Gr, Gb = get_GG_position(raw_pattern)

    Gr0 = raw_imgs[0, Gr[0]::2, Gr[1]::2]
    Gb0 = raw_imgs[0, Gb[0]::2, Gb[0]::2]
    
    Gr1 = raw_imgs[1, Gr[0]::2, Gr[1]::2]
    Gb1 = raw_imgs[1, Gb[0]::2, Gb[1]::2]

    Gr2 = raw_imgs[2, Gr[0]::2, Gr[1]::2]
    Gb2 = raw_imgs[2, Gb[0]::2, Gb[1]::2]
    
    sharp = np.zeros((3,))
    sharp[0] = (cv2.Laplacian(Gr0, 2) + cv2.Laplacian(Gb0, 2)).sum()
    sharp[1] = (cv2.Laplacian(Gr1, 2) + cv2.Laplacian(Gb1, 2)).sum()
    sharp[2] = (cv2.Laplacian(Gr2, 2) + cv2.Laplacian(Gb2, 2)).sum()

    # Select reference frame ID using sharpness value
    ref_frame_id = np.argmax(sharp)

    return ref_frame_id


def create_scale_pyramid(raw_imgs):
    _, h, w = raw_imgs.shape
    h = h // 2
    w = w // 2

    img_lvls = []

    # Create the level zero of the scale pyramid: it is average of all
    # Bayer pattern
    lvl0 = (raw_imgs[:, 1::2, 0::2] + 
            raw_imgs[:, 0::2, 0::2] + 
            raw_imgs[:, 1::2, 1::2] + 
            raw_imgs[:, 0::2, 1::2]) / 4

    # Pad zeros if it is not a multiple of 32, future downsampling makes sure it is 0
    lvl0 = np.pad(lvl0,((0, 0),(0, 32 - (h % 32)),(0,32 - (w % 32))), mode='edge')

    img_lvls.append(lvl0)

    # Extract shape of every L0 frame
    num_tiles, h, w = lvl0.shape


    # create gaussian pyramid with L0 -> L1 (down 2x wrt L0) -> L2 (down 8x wrt L0) -> L3 (down 32x wrt L0)
    lvl1 = np.zeros((num_tiles, h // 2, w // 2))
    lvl2 = np.zeros((num_tiles, h // 8, w // 8))
    lvl3 = np.zeros((num_tiles, h // 32, w // 32))

    # Create each level of pyramid 
    for i in range(num_tiles):
        frame = lvl0[i,:,:]
        frame = cv2.pyrDown(frame)
        lvl1[i,:,:] = frame
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        lvl2[i,:,:] = frame
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        lvl3[i,:,:] = frame

    # Appending other levels of the pyramids
    img_lvls.append(lvl1)
    img_lvls.append(lvl2)
    img_lvls.append(lvl3)

    return img_lvls

def align_images(ref_img, raw_imgs, use_temp=True):

    img_lvls = create_scale_pyramid(raw_imgs)

    return 0

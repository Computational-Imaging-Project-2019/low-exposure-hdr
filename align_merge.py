import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy

def get_GG_position(raw_pattern):
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

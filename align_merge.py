import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy
import helper

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
    lvl0 = np.pad(lvl0,((0, 0),(0, 32 - (h % 32)),(0, 32 - (w % 32))), mode='edge')

    img_lvls.append(lvl0)

    # Extract shape of every L0 frame
    num_tiles, h, w = lvl0.shape


    # create gaussian pyramid with L0 -> L1 (down 2x wrt L0) -> L2 (down 8x wrt L0) -> L3 (down 32x wrt L0)
    lvl1 = np.zeros((num_tiles, h // 2, w // 2))
    lvl2 = np.zeros((num_tiles, h // 8, w // 8))
    lvl3 = np.zeros((num_tiles, h // 32, w // 32))

    # Create each level of pyramid 
    for i in range(num_tiles):
        frame = lvl0[i, :, :]

        # L1
        frame = cv2.pyrDown(frame)
        lvl1[i, :, :] = frame

        # L2
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        lvl2[i, :, :] = frame

        # L3
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        lvl3[i, :, :] = frame

    # Appending other levels of the pyramids
    img_lvls.append(lvl1)
    img_lvls.append(lvl2)
    img_lvls.append(lvl3)

    return img_lvls

def align_level_3(ref_id, img_lvls):
    num_tiles = len(img_lvls[3])

    L3_size = img_lvls[3][0, ...].shape

    ref_tile = img_lvls[3][ref_id, ...] # gets the reference tile

    # Find padding size to make it perfect 8x8 blocks
    pad_rows = 8 - (L3_size[0] % 8)
    pad_cols = 8 - (L3_size[1] % 8)

    # Pad to make the it perfect 8x8 blocks
    ref_tile = np.pad(ref_tile, ((0, pad_rows), (0, pad_cols)), 'edge')
    L3_size = ref_tile.shape


    # search: 4 pixel left/top shift, 4 pixel right/bottom shift (9x9 search radius)
    ref_tile = np.pad(ref_tile, ((4, 4), (4, 4)), 'edge')

    align_shifts = []

    # Find alignment shift for each tile
    for tile_id in range(num_tiles):
        # Get the tile to be aligned
        tile = img_lvls[3][tile_id, ...]

        # Pad to make the it perfect 8x8 blocks
        tile = np.pad(tile, ((0, pad_rows), (0, pad_cols)), 'edge')

        err_stack = []

        # Find errors around 4x4 search radius
        for i in range(9):
            for j in range(9):
                # SSD of the overlapping parts
                err = (ref_tile[i : i + L3_size[0], j : j + L3_size[1]] - tile) ** 2

                # Border cases: Zero out the beyond border errors?
                # if i < 4:
                #     err[:i, :] = 0

                # if j < 4:
                #     err[:, :j] = 0

                # if i > 4:
                #     err[i + L3_size[0]:, :] = 0

                # if j > 4:
                #     err[:, j + + L3_size[1]:] = 0

                # Take sum of all l2 norm errors for each 8x8 block
                err = err.reshape(L3_size[0] // 8, 8, L3_size[1] // 8, 8).sum(axis=(1, 3))

                err_stack.append(err)

        # stack as numpy array
        err_stack = np.stack(err_stack)

        # Find the minimum error locations
        min_locs = np.argmin(err_stack, axis=0)

        # Keep shifts as (2, r, c)
        shifts = np.asarray(np.unravel_index(min_locs, (9, 9)))

        align_shifts.append(shifts)
    
    return np.stack(align_shifts)


        
                
                
        
        





def align_images(ref_id, raw_imgs, use_temp=True):

    img_lvls = create_scale_pyramid(raw_imgs)


    # Align L3 (the smallest)
    L3_shifts = align_level_3(ref_id, img_lvls)

    print(L3_shifts.shape)


    return 0

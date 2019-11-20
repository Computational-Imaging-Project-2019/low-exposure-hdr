import os
import numpy as np 
import subprocess
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rawpy
import helper
import skimage.util

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
    """ 
    input: raw_images
    output: scale pyramid gaussian

    Converts a stack of raw images to a set of gaussian pyramids of 4 levels, which are L0, L1 (2x Down), L2 (4x Down), L3 (4x Down)
    """
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

        # Find padding size to make it perfect 16x16 blocks
    if L3_size[0] % 8 != 0:
        pad_rows = 8 - (L3_size[0] % 8)
    
    if L3_size[1] % 8 != 0:
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
                # l2 norm of diff of the overlapping parts
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

        # Make the shifts accurate as (4,4) corresponds to (0,0)
        shifts = shifts - 4

        align_shifts.append(shifts)
    
    return np.stack(align_shifts)

def align_level_2(ref_id, img_lvls, L3_shifts):
    num_tiles = len(img_lvls[2])

    L2_size = img_lvls[2][0, ...].shape

    ref_tile = img_lvls[2][ref_id, ...] # gets the reference tile

    # Find padding size to make it perfect 16x16 blocks
    if L2_size[0] % 16 != 0:
        pad_rows = 16 - (L2_size[0] % 16)
    
    if L2_size[1] % 16 != 0:
        pad_cols = 16 - (L2_size[1] % 16)

    # Pad to make the it perfect 8x8 blocks
    ref_tile = np.pad(ref_tile, ((0, pad_rows), (0, pad_cols)), 'edge')
    L2_size = ref_tile.shape

    # Scale the L3 shifts for L2 shifts (as it is a 4x Down)
    L3_shifts = 4 * L3_shifts

    # Repeat the motion vectors for each 16x16 block
    L2_init_shifts = np.repeat(np.repeat(L3_shifts, 2, axis = -1), 2, axis = -2)
    
    if L2_init_shifts.shape[-1] != (L2_size[-1] // 16):
        L2_init_shifts = L2_init_shifts[:, :, :, :(L2_size[-1] // 16)]

    if L2_init_shifts.shape[-2] != (L2_size[-2] // 16):
        L2_init_shifts = L2_init_shifts[:, :, :(L2_size[-2] // 16), :]

    # search: 4 pixel left/top shift, 4 pixel right/bottom shift (9x9 search radius)
    ref_tile = np.pad(ref_tile, ((4, 4), (4, 4)), 'edge')

    align_shifts = []

    # Find alignment shift for each tile
    for tile_id in range(num_tiles):
        # Get the tile to be aligned
        tile = img_lvls[2][tile_id, ...]

        # Pad to make the it perfect 16x16 blocks
        tile = np.pad(tile, ((0, pad_rows), (0, pad_cols)), 'edge')

        #  Convert the tile into blocks of 16x16
        tile_blocks = skimage.util.view_as_blocks(tile, (16, 16))

        # Create matrix for storing the shifts
        tile_shift = np.zeros((tile_blocks.shape[0], tile_blocks.shape[1], 2))

        # Find errors around 4x4 search radius
        for row_idx in range(tile_blocks.shape[0]):
            for col_idx in range(tile_blocks.shape[1]):

                # Extract initial shifts
                init_shift = L2_init_shifts[tile_id, :, row_idx, col_idx]

                # Extract the tile_patch of 16x16
                tile_patch = tile_blocks[row_idx, col_idx, :, :]

                # Offsets for getting patch from ref_tile
                row_offset = row_idx * 16
                col_offset = col_idx * 16

                err_patch = []
                
                # Search the 9x9 radius around the ref_patch after initial offset
                for i in range(9):
                    for j in range(9):
                        # Extract the ref_patch
                        ref_patch = ref_tile[row_offset + i + init_shift[0] : row_offset + i + init_shift[0] + 16, col_offset + j + init_shift[1]: col_offset + j + init_shift[1] + 16]

                        # If patch is outside the ref_tile, it won't be 16x16, hence keep the error for it as inf
                        # TODO: Is there somthing better that can be done?
                        if ref_patch.shape != (16, 16):
                            err_patch.append(np.inf)
                            continue

                        # l2 norm of diff of the overlapping parts
                        err = ((ref_patch - tile_patch) ** 2).sum()

                        err_patch.append(err)
                
                # Find minimum error location
                err_patch = np.asarray(err_patch)
                min_loc = np.argmin(err_patch)
                shift = np.unravel_index(min_loc, (9, 9))

                # Store the shifts for each tile
                tile_shift[row_idx, col_idx, 0] = init_shift[0] + shift[0]
                tile_shift[row_idx, col_idx, 1] = init_shift[1] + shift[1]

        # Make the shifts accurate as (4,4) corresponds to (0,0)
        tile_shift = tile_shift - 4

        align_shifts.append(tile_shift)
    
    return np.stack(align_shifts)

def align_level_1(ref_id, img_lvls, L2_shifts):
    num_tiles = len(img_lvls[1])

    L1_size = img_lvls[1][0, ...].shape

    ref_tile = img_lvls[1][ref_id, ...] # gets the reference tile

    # Find padding size to make it perfect 16x16 blocks
    pad_rows, pad_cols = 0, 0
    if L1_size[0] % 16 != 0:
        pad_rows = 16 - (L1_size[0] % 16)
    
    if L1_size[1] % 16 != 0:
        pad_cols = 16 - (L1_size[1] % 16)

    # Pad to make the it perfect 8x8 blocks
    ref_tile = np.pad(ref_tile, ((0, pad_rows), (0, pad_cols)), 'edge')
    L1_size = ref_tile.shape

    # Scale the L2 shifts for L1 shifts (as it is a 4x Down)
    L2_shifts = 4 * L2_shifts

    # Repeat the motion vectors for each 16x16 block
    L1_init_shifts = np.repeat(np.repeat(L2_shifts, 4, axis = 1), 4, axis = 2)

    if L1_init_shifts.shape[1] != (L1_size[0] // 16):
        L1_init_shifts = L1_init_shifts[:, :(L1_size[0] // 16), :, :]

    if L1_init_shifts.shape[2] != (L1_size[1] / 16):
        L1_init_shifts = L1_init_shifts[:, :, :(L1_size[1] // 16), :]

    # search: 4 pixel left/top shift, 4 pixel right/bottom shift (9x9 search radius)
    ref_tile = np.pad(ref_tile, ((4, 4), (4, 4)), 'edge')

    align_shifts = []

    # Find alignment shift for each tile
    for tile_id in range(num_tiles):
        # Get the tile to be aligned
        tile = img_lvls[1][tile_id, ...]

        # Pad to make the it perfect 16x16 blocks
        tile = np.pad(tile, ((0, pad_rows), (0, pad_cols)), 'edge')

        #  Convert the tile into blocks of 16x16
        tile_blocks = skimage.util.view_as_blocks(tile, (16, 16))

        # Create matrix for storing the shifts
        tile_shift = np.zeros((tile_blocks.shape[0], tile_blocks.shape[1], 2))

        # Find errors around 4x4 search radius
        for row_idx in range(tile_blocks.shape[0]):
            for col_idx in range(tile_blocks.shape[1]):

                # Extract initial shifts, make it int
                init_shift = L1_init_shifts[tile_id, row_idx, col_idx, :].astype(int)

                # Extract the tile_patch of 16x16
                tile_patch = tile_blocks[row_idx, col_idx, :, :]

                # Offsets for getting patch from ref_tile
                row_offset = row_idx * 16
                col_offset = col_idx * 16

                err_patch = []
                
                # Search the 9x9 radius around the ref_patch after initial offset
                for i in range(9):
                    for j in range(9):
                        # Extract the ref_patch
                        ref_patch = ref_tile[row_offset + i + init_shift[0] : row_offset + i + init_shift[0] + 16, col_offset + j + init_shift[1]: col_offset + j + init_shift[1] + 16]

                        # If patch is outside the ref_tile, it won't be 16x16, hence keep the error for it as inf
                        # TODO: Is there somthing better that can be done?
                        if ref_patch.shape != (16, 16):
                            err_patch.append(np.inf)
                            continue

                        # l2 norm of diff of the overlapping parts
                        err = ((ref_patch - tile_patch) ** 2).sum()

                        err_patch.append(err)
                
                # Find minimum error location
                err_patch = np.asarray(err_patch)
                min_loc = np.argmin(err_patch)
                shift = np.unravel_index(min_loc, (9, 9))

                # Store the shifts for each tile
                tile_shift[row_idx, col_idx, 0] = init_shift[0] + shift[0]
                tile_shift[row_idx, col_idx, 1] = init_shift[1] + shift[1]

        # Make the shifts accurate as (4,4) corresponds to (0,0)
        tile_shift = tile_shift - 4

        align_shifts.append(tile_shift)
    
    return np.stack(align_shifts)

def align_level_0(ref_id, img_lvls, L1_shifts):
    num_tiles = len(img_lvls[0])

    L0_size = img_lvls[0][0, ...].shape

    ref_tile = img_lvls[0][ref_id, ...] # gets the reference tile

    # Find padding size to make it perfect 16x16 blocks
    pad_rows, pad_cols = 0, 0
    if L0_size[0] % 16 != 0:
        pad_rows = 16 - (L0_size[0] % 16)
    
    if L0_size[1] % 16 != 0:
        pad_cols = 16 - (L0_size[1] % 16)

    # Pad to make the it perfect 16x16 blocks
    ref_tile = np.pad(ref_tile, ((0, pad_rows), (0, pad_cols)), 'edge')
    L0_size = ref_tile.shape

    # Scale the L1 shifts for L0 shifts (as it is a 2x Down)
    L1_shifts = 2 * L1_shifts

    # Repeat the motion vectors for each 16x16 block
    L0_init_shifts = np.repeat(np.repeat(L1_shifts, 2, axis = 1), 2, axis = 2)
    
    if L0_init_shifts.shape[1] != (L0_size[0] // 16):
        L0_init_shifts = L0_init_shifts[:, :(L0_size[0] // 16), :, :]

    if L0_init_shifts.shape[2] != (L0_size[1] / 16):
        L0_init_shifts = L0_init_shifts[:, :, :(L0_size[1] // 16), :]

    # search: 1 pixel left/top shift, 1 pixel right/bottom shift (3x3 search radius)
    ref_tile = np.pad(ref_tile, ((1, 1), (1, 1)), 'edge')

    align_shifts = []

    # Find alignment shift for each tile
    for tile_id in range(num_tiles):
        # Get the tile to be aligned
        tile = img_lvls[0][tile_id, ...]

        # Pad to make the it perfect 16x16 blocks
        tile = np.pad(tile, ((0, pad_rows), (0, pad_cols)), 'edge')

        #  Convert the tile into blocks of 16x16
        tile_blocks = skimage.util.view_as_blocks(tile, (16, 16))

        # Create matrix for storing the shifts
        tile_shift = np.zeros((tile_blocks.shape[0], tile_blocks.shape[1], 2))

        # Find errors around 4x4 search radius
        for row_idx in range(tile_blocks.shape[0]):
            for col_idx in range(tile_blocks.shape[1]):

                # Extract initial shifts, make it int
                init_shift = L0_init_shifts[tile_id, row_idx, col_idx, :].astype(int)

                # Extract the tile_patch of 16x16
                tile_patch = tile_blocks[row_idx, col_idx, :, :]

                # Offsets for getting patch from ref_tile
                row_offset = row_idx * 16
                col_offset = col_idx * 16

                err_patch = []
                
                # Search the 3x3 radius around the ref_patch after initial offset
                for i in range(3):
                    for j in range(3):
                        # Extract the ref_patch
                        ref_patch = ref_tile[row_offset + i + init_shift[0] : row_offset + i + init_shift[0] + 16, col_offset + j + init_shift[1]: col_offset + j + init_shift[1] + 16]

                        # If patch is outside the ref_tile, it won't be 16x16, hence keep the error for it as inf
                        # TODO: Is there somthing better that can be done?
                        if ref_patch.shape != (16, 16):
                            err_patch.append(np.inf)
                            continue

                        # l2 norm of diff of the overlapping parts
                        err = (abs(ref_patch - tile_patch)).sum()

                        err_patch.append(err)
                
                # Find minimum error location
                err_patch = np.asarray(err_patch)
                min_loc = np.argmin(err_patch)
                shift = np.unravel_index(min_loc, (3, 3))

                # Store the shifts for each tile
                tile_shift[row_idx, col_idx, 0] = init_shift[0] + shift[0]
                tile_shift[row_idx, col_idx, 1] = init_shift[1] + shift[1]

        # Make the shifts accurate as (4,4) corresponds to (0,0)
        tile_shift = tile_shift - 1

        align_shifts.append(tile_shift)
    
    return np.stack(align_shifts)


def align_images(ref_id, raw_imgs, use_temp=True):

    temp_exists = os.path.isfile("./temp/L0_shifts.npy")
    
    if (temp_exists == False) or (use_temp == False):
        # Create Gaussian scale pyramid
        img_lvls = create_scale_pyramid(raw_imgs)

        # Align L3 (the smallest)
        print("Aligning L3 ...")
        L3_shifts = align_level_3(ref_id, img_lvls)

        # Align L2 using L3 shifts
        print("Aligning L2 ...")
        L2_shifts = align_level_2(ref_id, img_lvls, L3_shifts)

        # Align L1 using L2 shifts
        print("Aligning L1 ...")
        L1_shifts = align_level_1(ref_id, img_lvls, L2_shifts)

        # Align L0 using L1 shifts
        print("Aligning L0 ...")
        L0_shifts = align_level_0(ref_id, img_lvls, L1_shifts)

        # Note: L0_shifts now refer to shifts for every 32x32 block of the padded raw_img frames

        # Save the L0 shifts
        np.save("./temp/L0_shifts.npy", L0_shifts)
    else:
        print("Temp shifts present...")

        # Load the L0 shifts and return it
        L0_shifts = np.load("./temp/L0_shifts.npy")

    return L0_shifts

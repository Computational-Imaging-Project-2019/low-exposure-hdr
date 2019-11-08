import os 
import numpy as np 
import subprocess 
import argparse
import isp_helper
import time
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="HDR+: Low exposure images to HDR")
    parser.add_argument('-i', '--input', dest='input', default='./data/33TJ_20150612_201525_012', help='Enter the directory where the files are present')
    parser.add_argument('-o', '--output', dest='out', default='./results/', help='Enter the directory where the output files should be put')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,default=True, help='If 1, it will inform the steps on the console')
    parser.add_argument('-t', '--temp', dest='temp', type=int,default=True, help='If 1, it will use the temporary directory "./temp" to load variables, if that is not present, it will create one')

    args = parser.parse_args()
    return args

def create_temp_dir():
    """ Creates a temporary directory to save all the temp files
    such that it can be loaded immediately"""
    temp_path = "./temp"
    exists = os.path.exists(temp_path)
    if not exists:
        os.mkdir(temp_path)

if __name__ == "__main__":
    
    start_time = time.time()

    # Get input arguments
    args = get_args()

    # create temp directory
    create_temp_dir()

    # Create the raw files
    raw_imgs = isp_helper.get_raw(args.input, args.temp, args.verbose)

    # STEP 1: ALIGN
    # select reference frame
    Gb0 = raw_imgs[0, 1::2, 0::2]
    Gb1 = raw_imgs[1, 1::2, 0::2]
    Gb2 = raw_imgs[2, 1::2, 0::2]
    Gr2 = raw_imgs[2, 0::2, 1::2]
    Gr1 = raw_imgs[1, 0::2, 1::2]
    Gr0 = raw_imgs[0, 0::2, 1::2]
    sharp = np.zeros((3,))
    sharp[0] = np.sum(cv2.Laplacian(Gr0,2)+cv2.Laplacian(Gb0,2))
    sharp[1] = np.sum(cv2.Laplacian(Gr1,2)+cv2.Laplacian(Gb1,2))
    sharp[2] = np.sum(cv2.Laplacian(Gr2,2)+cv2.Laplacian(Gb2,2))
    ref_frame_id = np.argmax(sharp)
    # average out Bayer BGGR values
    raw_imgs = raw_imgs.astype('double')
    frames = (raw_imgs[:,1::2,0::2]+raw_imgs[:,0::2,0::2]+raw_imgs[:,1::2,1::2]+raw_imgs[:,0::2,1::2])/4
    # create gaussian pyramid
    # align level1
    # align level2
    # align level3
    # align level4
    
    # get raw object to indentify image parameters
    raw_obj = isp_helper.get_raw_object(args.input)

    end_time = time.time()
    print("Total time taken: {}".format(end_time - start_time))


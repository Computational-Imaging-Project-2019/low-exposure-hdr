import os 
import numpy as np 
import subprocess 
import argparse
import isp_helper
import time

def get_args():
    parser = argparse.ArgumentParser(description="HDR+")
    parser.add_argument('-i', '--input', dest='input', default='./data/33TJ_20150612_201525_012', help='Enter the directory where the files are present')
    parser.add_argument('-o', '--output', dest='out', default='./results/', help='Enter the directory where the output files should be put')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, help='If 1, it will inform the steps on the console')
    parser.add_argument('-s', '--skip_raw', dest='skip_raw', default=0, help='If 1, it will skip the raw file creation, assuming it is already present')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    start_time = time.time()

    # Get input arguments
    args = get_args()

    # Create the raw files
    raw_imgs = isp_helper.get_raw(args.input, args.skip_raw)

    end_time = time.time()
    print("Total time taken: {}".format(start_time - end_time))


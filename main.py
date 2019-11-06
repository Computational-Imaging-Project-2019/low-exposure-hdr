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
    parser.add_argument('-t', '--temp', dest='temp', default=1, help='If 1, it will use the temporary directory "./temp" to load variables, if that is not present, it will create one')

    args = parser.parse_args()
    return args

def create_temp_dir():
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
    raw_imgs = isp_helper.get_raw(args.input, args.temp)


    end_time = time.time()
    print("Total time taken: {}".format(end_time - start_time))


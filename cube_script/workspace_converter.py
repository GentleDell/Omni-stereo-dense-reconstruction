#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:47:56 2019

@author: zhantao
"""

import os
import glob
import subprocess
from shutil import rmtree

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam360 import Cam360
from colmap2mvsnet import convert_from_txt
from workspace_helper import check_path_exist, create_workspace_from_cam360_list

"""
1.Convert input/output data of our project to the required format for the MVSNet 
and OneNet.

2.Generate testing data for the MVSNet and OneNet to compare their performances.

3.MVSNet converter is adapted from colmap2mvsnet.py from MVSNet

"""

NUM_OF_VIEWS = 4

ONENET_BLOCK_SIZE = 0.3
ONENET_PIX_DROP_RATIO = 0.8
ONENET_PIX_DWON_RATIO = 0.5


def gen_test_OneNest(source_folder: str, target_folder: str, input_extension: str='png', output_extension: str='jpg'):
    '''
        It applies linear transformation to images in the given source_folder to 
        obatin training and testing data for the OneNet.
        
        In this codes, only center block, pixelwise inpainting and super resolution 
        are implimented.
    '''
    image_list = read_images(source_path = source_folder, extension=input_extension)
    
    check_path_exist(target_folder)

    for path in image_list:        
        img_name= path.split('/')[-1].split('.')[0]
        
        block_img, pixel_img, compressed_img = cv2.imread(path), cv2.imread(path), cv2.imread(path)
        h, w = pixel_img.shape[:2]
        
        block_img[ np.floor(h*(1-ONENET_BLOCK_SIZE)/2).astype(int): np.floor(h*(1+ONENET_BLOCK_SIZE)/2).astype(int), 
                   np.floor(w*(1-ONENET_BLOCK_SIZE)/2).astype(int): np.floor(w*(1+ONENET_BLOCK_SIZE)/2).astype(int), 
                   :] = 0  
        pixel_img[ np.random.choice(h, np.floor(ONENET_PIX_DROP_RATIO*h*w).astype(int)), 
                   np.random.choice(w, np.floor(ONENET_PIX_DROP_RATIO*h*w).astype(int)),
                   :] = 0
        compressed_img = cv2.resize(compressed_img, dsize=(np.floor(w/2).astype(int), np.floor(h/2).astype(int)), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(target_folder+'/{:s}_BLOCK.{:s}'.format(img_name, output_extension), block_img)
        cv2.imwrite(target_folder+'/{:s}_PIXEL.{:s}'.format(img_name, output_extension), pixel_img)
        cv2.imwrite(target_folder+'/{:s}_COMPR.{:s}'.format(img_name, output_extension), compressed_img)
    
    print('''ATTENTION: copy all files to /where_the_onenet_is/datasets/celeb-1m/test/''')
    
    
def read_images(source_path: str, extension: str='png'):
    
    # check path existence
    if not os.path.exists(source_path):
        raise ValueError("{:s} does not exist.".format(source_path))
    
    # read images recursively
    image_files = []
    for dir, _, _, in os.walk(source_path):
        filenames = glob.glob( os.path.join(dir, '*.{:s}'.format(extension)))  
        image_files += filenames
    
    # check the num of images
    if len(image_files) == 0:
        raise ValueError("{:s} does not contain required image.".format(source_path))
    
    return image_files


def gen_input_MVSNet_from_colmap(source_folder: str, target_folder: str, patchmatch_path: str, 
                                 enable_cubic_projection: bool=False, extrinsic_mat: np.array=np.array([]), 
                                 ref_image_index: int=5, enable_selection: bool=False):
    '''
        It converts the given colmap workspace or the folder containing testing images
        to mvsnet workspace format. Some functions are adapted from mvsnet.
    '''
    
    temp_dir = source_folder
    
    # If the source folder is not a colmap workspace and only contains images,
    # decompose 360 images in this folder and save for colmap.
    if enable_cubic_projection:   
        
        if extrinsic_mat.size == 0:
            raise ValueError("Camera extrinsic parameters should be provided.")
        elif extrinsic_mat.size % 12 != 0:
            raise ValueError("Input ERROR! Input poses should be an nx12 matrix")
        else:
            poses     = extrinsic_mat.reshape(-1,12)
            rotations = poses[:,:9].reshape(-1,3,3)
            translations = poses[:,9:]
            
        for t in range(translations.shape[0]):
            # inv([R,t]) = [R', -R'*t];   R,t: rotation and translation from world to local 
            translations[t] = - rotations[t,:,:].dot(translations[t])
        
        image_list= sorted(glob.glob(os.path.join(source_folder, '*.png')))
        assert(len(image_list) == poses.shape[0])    # make sure the number of poses and images are the same.
        
        cam360_list = [] 
        for ind, filename in enumerate(image_list):
            # load omnidirectional images
            Omni_img = np.flip(cv2.imread(filename), axis=2)   
            Omni_img = Omni_img/np.max(Omni_img)
            # create a Cam360 object
            Omni_obj = Cam360(rotation_mtx = rotations[ind,:,:], 
                              translation_vec=translations[ind], 
                              height = Omni_img.shape[0], 
                              width  = Omni_img.shape[1], 
                              channels = Omni_img.shape[2], 
                              texture= Omni_img)
            cam360_list.append(Omni_obj)
        
        temp_dir = os.path.join(target_folder,'temp')
        check_path_exist(temp_dir)
        create_workspace_from_cam360_list(cam_list=cam360_list, refimage_index=ref_image_index, number_of_views = NUM_OF_VIEWS,
                                          work_dir = temp_dir, view_selection=enable_selection)
    
    # Use colmap image_undistorter to reorganize folders into colmap format 
    source_folder = arange_folder(temp_dir, patchmatch_path, NUM_OF_VIEWS)
    
    # Convert colmap formate to mvsnet formatsss
    for folder in glob.glob(source_folder):
        folder_name = folder.split('/')[-1]
        convert_from_txt(folder, os.path.join(target_folder, folder_name))
    
    # Clean temp files
    if enable_cubic_projection:
        rmtree(temp_dir)
    
    
def arange_folder(folder: str, PMS_path: str, num_views: int=4):
    
    for view in range(num_views):
        
        input_path = os.path.join(folder, "cubemaps/parameters/view" + str(view))
        output_path= os.path.join(folder, "patch_match_ws/view" + str(view))
        image_path = os.path.join(folder, "cubemaps/view" + str(view))
        
        check_path_exist(output_path)
        
        # undistort images and orgnize workspace for dense reconstruction
        command = PMS_path + \
                  " image_undistorter" + \
                  " --image_path="  + image_path + \
                  " --input_path="  + input_path + \
                  " --output_path=" + output_path
        CM = subprocess.Popen(command, shell=True)
        CM.wait()
        
    return os.path.join(folder, "patch_match_ws/*")
    
    
    
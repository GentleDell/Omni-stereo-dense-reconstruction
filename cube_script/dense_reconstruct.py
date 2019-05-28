#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:33:28 2019

@author: zhantao
"""
import os
import glob
import argparse

import cv2 
import numpy as np
from cam360 import Cam360
from workspace_helper import dense_from_cam360list


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_image"  ,  type=str, default='../images', help="where the 360 imagses are stored")
parser.add_argument("--patchmatch_path",  type=str, default='./colmap', help="Where is the exectable file of patch matching stereo GPU")
parser.add_argument("--workspace"      ,  type=str, default='../workspace',  help="Where to store the workspace") 
parser.add_argument("--reference_image",  type=int, default=4,  help="Which image is the used as the reference image to create the world coordinate")
parser.add_argument("--views_for_synthesis",  type=int, default=4,  help="The number of views to synthesize the 360 depthmap; only 4 and 6 are supported") 
parser.add_argument("--use_colmap",  type=bool, default=False,  help="Use orignal colmap or the modified PatchMatchingStereoGPU adapted from colmap") 
parser.add_argument("--pose_list" ,  nargs='+', default=['4'],  help="The number of views to synthesize the 360 depthmap") 


def main():
    
    args = parser.parse_args()
    print('----------------------------------------')
    print('FLAGS:')
    for arg in vars(args):
        print( '{:<20}: {:<20}'.format(arg, str(getattr(args, arg))) )
    print('----------------------------------------')
    
    
    if args.pose_list[0] == '4' and len(args.pose_list) == 1:
        translations = np.array([[-4, -4, 1], [-4, 0, 1], [-4, 4, 1],
                                 [ 0, -4, 1], [0, 0, 1] , [ 0, 4, 1],
                                 [ 4, -4, 1], [4, 0, 1] , [ 4, 4, 1]])
    elif args.pose_list[0] == '1' and len(args.pose_list) == 1:
        translations = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1],
                                 [ 0, -1, 1], [0, 0, 1] , [ 0, 1, 1],
                                 [ 1, -1, 1], [1, 0, 1] , [ 1, 1, 1]])
    else:
        poses = args.pose_list[0].split(',')
        if len(poses) % 3 != 0:
            raise ValueError("Input ERROR! Input translatetions should be an nx3 matrix")
        else:
            poses = [float(t) for t in poses]
            translations = np.array(poses).reshape(-1,3)
            print(translations)
            
            
    cam360_list = []
    for ind, filename in enumerate(sorted(glob.glob(os.path.join(args.path_to_image, '*.png')))):
        # load omnidirectional images
        Omni_img = np.flip(cv2.imread(filename), axis=2)   
        Omni_img = Omni_img/np.max(Omni_img)
        # create a Cam360 object
        Omni_obj = Cam360(rotation_mtx = np.eye(3), 
                          translation_vec=translations[ind], 
                          height = Omni_img.shape[0], 
                          width = Omni_img.shape[1], 
                          channels = Omni_img.shape[2], 
                          texture= Omni_img)
        cam360_list.append(Omni_obj)
        
    dense_from_cam360list(cam360_list, 
                          workspace = args.workspace,
                          reference_image = args.reference_image,  
                          patchmatch_path = args.patchmatch_path, 
                          views_for_synthesis = args.views_for_synthesis,
                          use_colmap = args.use_colmap)


if __name__ == '__main__':
    main()
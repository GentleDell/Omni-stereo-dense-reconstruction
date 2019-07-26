#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:33:28 2019

@author: zhantao
"""
import os
import glob
import pickle # for synthesis
import argparse

import cv2 
import numpy as np
from cam360 import Cam360
from workspace_helper import dense_from_cam360list


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_image"  ,  type=str,  default='../data_demo/dataset', help="Where the 360 imagses are stored")

parser.add_argument("--patchmatch_path",  type=str,  default='../colmap', help="Where is the exectable file of patch matching stereo GPU")

parser.add_argument("--workspace"      ,  type=str,  default='../data_demo/workspace',  help="Where to store the workspace") 

parser.add_argument("--reference_view" ,  type=int , default=4,  help="The index of the reference view. Only works when view_selection is disabled.") 

parser.add_argument("--view_selection" , default=True, action='store_true', help="Select views for dense reconstruction") 

parser.add_argument("--views_for_depth",  type=int , default=4,  help="The number of views to synthesize the 360 depthmap; only 4 and 6 are supported") 

parser.add_argument("--gpu_index",  type=int , default=2,  help="The index of GPU to run the Patch Matching") 

parser.add_argument("--pose_list"      ,  nargs='+', default=['4'],  help="A list of poses corresponding to the given images; e.g. R1,t1,R2,t2,... (from world to local)") 


def main():
    
    args = parser.parse_args()
    print('----------------------------------------')
    print('FLAGS:')
    for arg in vars(args):
        print( '{:<20}: {:<20}'.format(arg, str(getattr(args, arg))) )
    print('----------------------------------------')
    
    if len(args.pose_list[0]) == 1:
        rotations = np.eye(3)
        rotations = np.repeat(rotations[:,:,None], 9, axis=2)
        translations = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1],
                                 [ 0, -1, 1], [0, 0, 1] , [ 0, 1, 1],
                                 [ 1, -1, 1], [1, 0, 1] , [ 1, 1, 1]])
        translations = float(args.pose_list[0]) * translations    # scale
    else:
        poses = args.pose_list[0].split(',')
        if len(poses) % 12 != 0:
            raise ValueError("Input ERROR! Input poses should be able to be transposed to an nx12 matrix")
        else:
            poses     = np.array([float(t) for t in poses]).reshape(-1,12)
            rotations = np.swapaxes(poses[:,:9].reshape(3,3,-1), 0, 2)
            translations = poses[:,9:]
            
    for t in range(translations.shape[0]):
        # inv([R,t]) = [R', -R'*t];   R,t: rotation and translation from world to local 
        translations[t] = - rotations[:,:,t].dot(translations[t])
        
    print(rotations[:,:,0], translations)
            
            
    #################################################################################


    cam360_list = []
    for ind, filename in enumerate(sorted(glob.glob(os.path.join(args.path_to_image, '*.png')))):
        # load omnidirectional images
        Omni_img = np.flip(cv2.imread(filename), axis=2)   
        Omni_img = Omni_img/np.max(Omni_img)
        # create a Cam360 object
        Omni_obj = Cam360(rotation_mtx = rotations[:,:,ind], 
                          translation_vec=translations[ind], 
                          height = Omni_img.shape[0], 
                          width  = Omni_img.shape[1], 
                          channels = Omni_img.shape[2], 
                          texture= Omni_img)
        cam360_list.append(Omni_obj)
        
    cam360_list = dense_from_cam360list(cam360_list, 
                                        workspace = args.workspace,
                                        patchmatch_path = args.patchmatch_path, 
                                        reference_view  = args.reference_view,
                                        views_for_depth = args.views_for_depth,
                                        use_view_selection = args.view_selection,
                                        gpu_index = args.gpu_index)
    
    # save data for view synthesis
    pickle_out = open(os.path.join(args.workspace,"cam360.pickle"),"wb")
    pickle.dump(cam360_list, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:57:57 2019

@author: zhantao
"""
import os
import glob

import cv2
import json
import numpy as np 
import matplotlib.pyplot as plt

from cam360 import Cam360
from spherelib import rotation_mtx

_INVALID_DEPTH_STANFORD_ = 65535
_UNIT_DEPTH_STANFORD_    = 1/512

def stanford_3D_dataset( path_to_scene: str ):
    '''
        now the dataset is reorganized manually, will adapted for the original 
        dataset in the future
    '''
    file_list = sorted(glob.glob(os.path.join(path_to_scene, '*_rgb.png')))
    
    if len(file_list) == 0:
        raise ValueError('Invalid path to stanford dataset')
    
    cam360_list = []
    for file in file_list:
        
        img_name  = '_'.join( file.split('/')[-1].split('_')[:4] )
        pose_json = glob.glob(os.path.join(path_to_scene, img_name+'*.json'))[0]
        depth_GT  = glob.glob(os.path.join(path_to_scene, img_name+'*_depth.png'))[0]
        
        image360 = np.flip(cv2.imread(file), axis=2)   
        image360 = image360/np.max(image360)

        depth_GT = cv2.imread(depth_GT, -1)                   # read as 16bits grayscale image
        depth_GT[depth_GT == _INVALID_DEPTH_STANFORD_] = 0    # set invalid depth to 0
        depth_GT = depth_GT * _UNIT_DEPTH_STANFORD_           # convert to real depth
        
        with open(pose_json, 'r') as f:
            pose360= json.loads(f.read())
#        rotation = rotation_mtx( np.array(pose360['final_camera_rotation']), order = (0,1,2) )
#        translation = np.array(pose360['camera_location'])
            
        pose = np.array(pose360['camera_rt_matrix'])
        rotation = pose[:3,:3]
        translation = pose[:,3]
        
        cam = Cam360(rotation_mtx = rotation, translation_vec = translation, 
                     height = image360.shape[0], width = image360.shape[1], channels = image360.shape[2], 
                     texture= image360)
        cam360_list.append(cam)
        
    return cam360_list
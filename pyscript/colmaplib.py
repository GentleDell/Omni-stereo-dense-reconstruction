#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  19 18:39:57 2019

@author: zhantao
"""
import os
import glob

import cv2
import numpy as np

from cam360 import Cam360
from cubicmaps import CubicMaps

def create_workspace(image_dir: str='', file_suffix: str='png', work_dir: str='./', 
                     is_depth: bool = False, radius_depth: bool = False, cam_para: list = None):
        """
            It projects all omnidirectional images under the given directory to cubic maps. 
            Then it collects cubic maps according to the view (back/front/laft/right) and 
            saves diffetent views to the corresponding folders (/work_dir/view0, /work_dir/view1, etc.).
            
            Parameters
            ----------    
            image_dir : str
                Path to the omnidirectional images;
                
            file_suffix : str
                The format of omnidirectional images;
                
            work_dir :
                Where to save cubic images;
                
            is_depth: bool
                Whether the images to be load are depthmaps (.exr file);
                
            radius_depth: bool
                Whether to save radius depthmap or the euclidean depth;
                only used for comparing the estimated depth map with the ground truth;  
            
            cam_para: list
                List of camera parameters, containing 3 parameters: 
                [focal length x, focal length y, camera center on rows, camera center on columns]
    
            Returns
            -------
            Omni_image : np.array
                An omnidirectional image generated from the given 6 cubic images.
            
            Examples
            --------
            >>> create_workspace(image_dir='./data', file_suffix='png', work_dir = './cubic_maps')
        """
        if len(file_suffix) <= 0:
                raise ValueError("Input ERROR! Invalid file suffix") 
                   
        cubemap_obj = CubicMaps()
        
        file_pattern = '*.'+file_suffix
        for filename in glob.glob(os.path.join(image_dir, file_pattern)):
            
            prefix = filename.split(sep='/')[-1][:-4]
            
            if is_depth==False:
                # load omnidirectional images
                Omni_img = np.flip(cv2.imread(filename), axis=2)   
                Omni_img = Omni_img/np.max(Omni_img)
                
                Omni_obj = Cam360(rotation_mtx = np.eye(3), translation_vec=np.zeros([3,1]), 
                                  height = Omni_img.shape[0], width = Omni_img.shape[1], channels = Omni_img.shape[2], 
                                  texture= Omni_img)
                                
                cubemap_obj.sphere2cube(Omni_obj, resolution=(512,512))
                
                for ind in range(len(cubemap_obj.cubemap)):
                    directory = work_dir + '/view' + str(ind) + '/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    cubemap_obj.save_cubemap(path = directory, prefix = prefix, index=[ind])     
                
            else:
                # load omnidirectional depthmap
                Omni_dep = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
                
                Omni_obj = Cam360(rotation_mtx = np.eye(3), translation_vec=np.zeros([3,1]), 
                                  height = Omni_dep.shape[0], width = Omni_dep.shape[1], channels = 1, 
                                  depth = Omni_dep)
            
                cubemap_obj.sphere2cube(Omni_obj, resolution=(512,512), is_depth=is_depth)
                    
                for ind in range(len(cubemap_obj.depthmap)):
                    directory = work_dir + '/view' + str(ind) + '/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    if radius_depth:
                        cubemap_obj.save_cubedepth(path = directory, prefix = prefix, index=[ind], 
                                                   dist_to_radius=True, camera_para=cam_para)     
                    else:
                        cubemap_obj.save_cubedepth(path = directory, prefix = prefix, index=[ind])     
           
def create_camera_model( intrinsic_para: list=[], num_camera: int = 2, same_camera: bool = True):
    if 
#    
#def create_imagedb():
        
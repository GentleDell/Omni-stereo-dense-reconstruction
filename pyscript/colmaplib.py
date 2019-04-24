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
from scipy.spatial.transform import Rotation

VIEW_ROT = np.array([[[-1,0,0], [0,0,-1], [0,-1,0]],
                     [[ 0,1,0], [0,0,-1], [-1,0,0]],
                     [[ 1,0,0], [0,0,-1], [0, 1,0]],
                     [[0,-1,0], [0,0,-1], [ 1,0,0]],
                     [[ 1,0,0], [0, 1,0], [0,0, 1]],
                     [[ 1,0,0], [0,-1,0], [0,0,-1]]])

def create_workspace(image_dir: str='', file_suffix: str='png', work_dir: str='./', 
                     camera_parameters: list = None, reference_pose: np.array = None):
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
        
        Examples
        --------
        >>> create_workspace(image_dir='./data', file_suffix='png', work_dir = './cubic_maps')
    """
    
    if len(file_suffix) <= 0:
            raise ValueError("Input ERROR! Invalid file suffix") 
    # initialize a CubeMaps object
    cubemap_obj = CubicMaps()
    # initialize a list of flags to record whether the camera model for the 
    # 6 cubemaps have been written to the file
    if camera_parameters:
        flag_cam_model = [True]*6
    
    file_pattern = '*.'+file_suffix
    for filename in glob.glob(os.path.join(image_dir, file_pattern)):
        prefix = filename.split(sep='/')[-1][:-4]
        # load omnidirectional images
        Omni_img = np.flip(cv2.imread(filename), axis=2)   
        Omni_img = Omni_img/np.max(Omni_img)
        # create a Cam360 object
        Omni_obj = Cam360(rotation_mtx = np.eye(3), translation_vec=np.zeros([3,1]), 
                          height = Omni_img.shape[0], width = Omni_img.shape[1], channels = Omni_img.shape[2], 
                          texture= Omni_img)
        # project the omnidirectional image to 6 cubemaps
        cubemap_obj.sphere2cube(Omni_obj, resolution=(512,512))
        # save the cubemaps
        for ind in range(len(cubemap_obj.cubemap)):
            directory = work_dir + '/view' + str(ind) + '/'
            # create the directory if not exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            # save cubemaps to the folder
            cubemap_obj.save_cubemap(path = directory, prefix = prefix, index=[ind])
            # if required to create camera model file 
            if camera_parameters is not None and flag_cam_model[ind]:
                # write camera model to the given directory
                create_camera_model(path=directory, camera_para = camera_parameters, camera_size=cubemap_obj.cubemap[ind].shape[:2])
                # suggest the model has been written
                flag_cam_model[ind] = False
            
            if reference_pose is not None:
                if len(reference_pose) == 7:
                    create_imagedb( path=directory, name_list = [prefix], camera_id = 1, ref_pose=reference_pose, view_ind=ind)
                else:
                    raise ValueError('Input ERROR! Invalid reference pose, it should be a quaternion followed by a translation')
                    
                        

def create_depth_workspace(image_dir: str='', file_suffix: str='exr', work_dir: str='./',
                           radius_depth: bool = False, cam_para: list = None):
    """
        It projects all omnidirectional depthmaps under the given directory to cubic maps. 
        Then it collects cubic depthmaps according to the view (back/front/laft/right) and 
        saves diffetent views to the corresponding folders (/work_dir/view0, /work_dir/view1, etc.).
        
        Parameters
        ----------    
        image_dir : str
            Path to the omnidirectional images;
            
        file_suffix : str
            The format of omnidirectional images;
            
        work_dir :
            Where to save cubic images;
            
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
        
        # load omnidirectional depthmap
        Omni_dep = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        
        Omni_obj = Cam360(rotation_mtx = np.eye(3), translation_vec=np.zeros([3,1]), 
                          height = Omni_dep.shape[0], width = Omni_dep.shape[1], channels = 1, 
                          depth = Omni_dep)
    
        cubemap_obj.sphere2cube(Omni_obj, resolution=(512,512), is_depth = True)
            
        for ind in range(len(cubemap_obj.depthmap)):
            directory = work_dir + '/view' + str(ind) + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            if radius_depth:
                cubemap_obj.save_cubedepth(path = directory, prefix = prefix, index=[ind], 
                                           dist_to_radius=True, camera_para=cam_para)     
            else:
                cubemap_obj.save_cubedepth(path = directory, prefix = prefix, index=[ind])     
                
                        
def create_camera_model( path: str = './', camera_para: list = None, camera_size: tuple = None):
    """
        It creates a file containing several camera models for colmap.
        
        Parameters
        ----------    
        path: str
            Path to save the file;
            
        camera_para: list of list
            This is a list of list. In each sublist, it contains parameters to be saved, including: [fx, fy, cx, cy].
            Different sublists represent different cameras.
            
        camera_size: tuple
            The size of images taken by the camera.
        
        Return:
        --------
        A .txt file containing camera parameters. Following is an example:
            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: 2
            1 PINHOLE 512 512 358.29301602 358.29301602 256 256
            2 PINHOLE 512 512 267.29301602 267.29301602 256 256
        
        Examples
        --------
        >>> create_camera_model(path='./', camera_para = [267,267,256,256], image_size=(512,512))
        
    """
    if camera_para is None or camera_size is None:
        raise ValueError("Input ERROR! Camera parameters or image size are not provided") 
    elif len(camera_size) != 2:
        raise ValueError("Input ERROR! Invalid image size")
    else:
        helper = '# Camera list with one line of data per camera: \n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n# Number of cameras: 1 \n'     
        with open(path + 'cameras.txt', "a+") as f:   # Opens file and casts as f 
            for camera_id in range(len(camera_para)):
                # Writing
                f.write(helper)
                f.write(str(camera_id+1) + ' PINHOLE ' + str(camera_size[0]) + ' ' + str(camera_size[1]))
                for para in camera_para[camera_id]:
                    f.write( ' ' + str(para))
                f.write('\n')
                f.close()         # Close file
        
        
def create_imagedb( path: str = './', name_list: list = None, camera_id: int = None, ref_pose: np.array = None, view_ind: int = None):
    '''
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        # Number of images: 2, mean observations per image: 2
        1 1 0 0 0 4 -1 0 1 view3l.png
        
        2 1 0 0 0 -4 -1 0 2 view3r.png
    '''
    if name_list is None or camera_id is None or ref_pose is None:
        raise ValueError("Input ERROR! One or more parameters are not provided") 
    elif len(ref_pose) != 7:
        raise ValueError("Input ERROR! Invalid reference pose")
    else:
        with open(path + 'images.txt', "a+") as f:
            for ind, image_name in enumerate(name_list):
                
                rotation, translation = pose_from_name(image_name, ref_pose, view_ind)
                
                f.write(str(ind+1))
                for quant in rotation:
                    f.write( ' ' + str(quant))
                for trans in translation:
                    f.write( ' ' + str(trans))
                f.write( ' ' + str(camera_id) + ' ' + image_name)
                f.write('\n')
                f.close()                                   # Close file
    

def pose_from_name(name:str, ref_pose: np.array, view_index: int) -> np.array:
    try:
        words = name.split(sep='_')
        # here Rotation use [x,y,z,w] but in colmap it uses [w,x,y,z]
        tocolmap = Rotation.from_dcm(VIEW_ROT[view_index,:,:])
        
        # if each image has it own rotation, then we need to parse the rotation here 
        rot_img = tocolmap.__mul__( Rotation.from_dcm( np.eye(3) ))
        trans_img = words[1:4]
        trans_img = np.array([int(trans) for trans in trans_img])
        
        
        # here Rotation use [x,y,z,w] but in colmap it uses [w,x,y,z]
        trans_ref = np.array(ref_pose[4:])
        quat_ref = [a for a in ref_pose[1:4]]
        quat_ref.append(ref_pose[0])
        rot_ref = tocolmap.__mul__(Rotation(quat_ref))
        
        rotation = Rotation.from_dcm(rot_ref.__mul__(rot_img).as_dcm()).as_quat()
        translation = rot_img.as_dcm().dot(trans_ref - trans_img) 
              
    except:
        raise ValueError('Input ERROR! Can not parse the given filename to poses')
    return rotation, translation
    
    
    
    
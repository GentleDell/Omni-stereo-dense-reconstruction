#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  19 18:39:57 2019

@author: zhantao
"""
import os
import glob
from typing import Optional

import cv2
import numpy as np

from cam360 import Cam360
from cubicmaps import CubicMaps
from scipy.spatial.transform import Rotation

'''
The below shows the transformations between the world coordinate used in Cam360
and the local camera coordinates used in the colmap as well as the 6 cubemaps.

In the colmap and cubemaps, the local camera coordinate system of an image is 
defined in the way that the x axis points to the right, the y axis to the bottom 
and the Z axis to the front as seen from the image.

The world coordinate is the same as the local camera coordinate of the 4th 
view (the top view).

                                        view2:[colmap] 
                                            z_2
                                             /
                                            /------- x_2
                                            |            
                                            | y_2           


                                        [world coorinate]
                 view1:[colmap]             Z                    view3:[colmap] 
                            x_1             |  /Y                     -------- z_3
                            /               | /                      /|
                           /                |/                      / |
                  z_1 ----|                 |--------X             /  |
                          |                                       /   |
                          |                                     x_3   y_3
                          y_1     
                                  view0:[colmap] 
                                     x_0 ----|
                                            /|
                                           / |
                                        z_0  y_0
'''
# the rotation matrixs of the 6 cubemaps
VIEW_ROT = np.array([[[-1,0,0], [0,0,-1], [0,-1,0]],
                     [[ 0,1,0], [0,0,-1], [-1,0,0]],
                     [[ 1,0,0], [0,0,-1], [0, 1,0]],
                     [[0,-1,0], [0,0,-1], [ 1,0,0]],
                     [[ 1,0,0], [0, 1,0], [0,0, 1]],
                     [[ 1,0,0], [0,-1,0], [0,0,-1]]])


def create_workspace(image_dir: str='', file_suffix: str='png', work_dir: str='./', 
                     camera_parameters: list = None, reference_pose: list = None):
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
            
        camera_parameters: list of list
            This is a list of list. In each sublist, it contains parameters to be saved, including: [fx, fy, cx, cy].
            Different sublists represent different cameras. 
            
        reference_pose: list -> [w, x, y, z, tx, ty, tz ]
            Quaternion and translation vector.
            The pose of the center camera (the camera in the center of the camera array).
            When using several images in a folder to reconstruct a scene, the colmap will locate each image(camera) by their
            relative poes to the center image(camera). So, to ceate a workspace for colmap, the relative poses are required.
            Therefore, the pose of the center image(camera) is required to be specified.
        
        Examples
        --------
        >>> create_workspace(image_dir='./image', file_suffix='png', work_dir = './ws', 
                             camera_parameters=[[267,267,256,256]], reference_pose=[1,0,0,0,0,0,1])
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
    image_counter = 0
    for filename in glob.glob(os.path.join(image_dir, file_pattern)):
        image_counter += 1
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
            para_directory = work_dir + '/paramters/view' + str(ind) + '/'
            # create the directory if not exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(para_directory):
                os.makedirs(para_directory)
            # save cubemaps to the folder
            cubemap_obj.save_cubemap(path = directory, prefix = prefix, index=[ind])
            # if required to create camera model file 
            if camera_parameters is not None and flag_cam_model[ind]:
                # create an empty file to represent 3d sparse models
                with open(para_directory + 'points3D.txt', "a+") as f: 
                    f.closed       
                # write camera model to the given directory
                create_camera_model(path=para_directory, camera_para = camera_parameters, camera_size=cubemap_obj.cubemap[ind].shape[:2])
                # suggest the model has been written
                flag_cam_model[ind] = False
            
            if reference_pose is not None:
                if len(reference_pose) == 7:
                    create_imagedb( path=para_directory, name_list = [ prefix + '_view' + str(ind) + '.' + file_suffix], 
                                    camera_id = 1, ref_pose=np.array(reference_pose), view_ind = ind, image_id = image_counter)
                else:
                    raise ValueError('Input ERROR! Invalid reference pose, it should be a quaternion followed by a translation')
                    
                        

def create_depth_workspace(image_dir: str='', file_suffix: str='exr', work_dir: str='./',
                           radius_depth: Optional[bool] = False, cam_para: Optional[list] = None):
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
    # initialize a CubeMaps object
    cubemap_obj = CubicMaps()
    
    file_pattern = '*.'+file_suffix
    
    for filename in glob.glob(os.path.join(image_dir, file_pattern)):
        prefix = filename.split(sep='/')[-1][:-4]
        
        # load omnidirectional depthmap
        Omni_dep = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        # create a Cam360 object
        Omni_obj = Cam360(rotation_mtx = np.eye(3), translation_vec=np.zeros([3,1]), 
                          height = Omni_dep.shape[0], width = Omni_dep.shape[1], channels = 1, 
                          depth = Omni_dep)
        # project the omnidirectional image to 6 cubemaps
        cubemap_obj.sphere2cube(Omni_obj, resolution=(512,512), is_depth = True)
        # save the cubemaps
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
            f.write(helper)
            for camera_id in range(len(camera_para)):
                # Writing
                f.write(str(camera_id+1) + ' PINHOLE ' + str(camera_size[0]) + ' ' + str(camera_size[1]))
                for para in camera_para[camera_id]:
                    f.write( ' ' + str(para))
                f.write('\n')
                f.close()         # Close file
        
        
def create_imagedb(path: str = './', name_list: list = None, 
                   camera_id: int = None, ref_pose: np.array = None, 
                   view_ind: int = None, image_id:int = None):
    '''
        It parses images' (cameras') poses from images' names and then creates an image database for colmap.
        
        Parameters
        ----------    
        path: str
            Path to save the file;
            
        name_list: list
            A list of images names;
            
        camera_id: int
            The camera id of the camera that captures these images given in the name_list;
            
        ref_pose: np.array
            The pose of the camera that capturing the center image.
            
        view_ind: int 
            The index of the current view: [ 0th:  back  |  1st:  left  |  2nd:  front  |  3rd:  right  |  4th:  top  |  5th:  bottom ]
            Different views have different rotation matrices.  
            
        image_id: int
            Used as the image_id in the image database.     
        
        Return:
        --------
        A .txt file containing an image database. The following is an example:
            # Image list with two lines of data per image:\n
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n
            #   POINTS2D[] as (X, Y, POINT3D_ID)\n
            # Number of images: 1\n
            1 1 0 0 0 4 -1 0 1 view3l.png
    '''
    
    if name_list is None or camera_id is None or ref_pose is None:
        raise ValueError("Input ERROR! One or more parameters are not provided") 
    elif len(ref_pose) != 7:
        raise ValueError("Input ERROR! Invalid reference pose")
    else:
        with open(path + 'images.txt', "a+") as f:
            for ind, image_name in enumerate(name_list):
                # parse the image names to poses
                rotation, translation = pose_from_name(image_name, ref_pose, view_ind)
                # record image id, pose, camera id as well the image name to the images.txt
                # in the colmap format
                f.write(str(image_id + ind))
                for quant in rotation:
                    f.write( ' ' + str(quant))
                for trans in translation:
                    f.write( ' ' + str(trans))
                f.write( ' ' + str(camera_id) + ' ' + image_name)
                f.write('\n\n')
                # Close file
                f.close()                                  
    

def pose_from_name(name:str, ref_pose: np.array, view_index: int):
    '''
        It parses the pose of an image/camera according to the image's name.
        
        Parameters
        ----------    
        name:str
            The name of the images;
            
        ref_pose: np.array
            The pose of the camera that capturing the center image; also called 'the pose of the reference image';
            
        view_ind: int 
            The index of the current view: [ 0th:  back  |  1st:  left  |  2nd:  front  |  3rd:  right  |  4th:  top  |  5th:  bottom ]
            Different views have different rotation matrices.  
        
        Return:
        --------
        quat_final: list
            The quaternion vector;
        
        translation: nd.array
            The translation vector
    '''
    try:
        tocolmap = VIEW_ROT[view_index,:,:]
        
    # parse the image name to the corresponding pose
        words = name.split(sep='_')
        # if each image has it own rotation, then we need to parse the rotation here 
        rot_img =  Rotation.from_dcm(np.eye(3))
        trans_img = words[1:4]
        trans_img = np.array([int(trans) for trans in trans_img])
        
    # convert the pose of the reference image
        trans_ref = np.array(ref_pose[4:])
        # here scipy.transformation.Rotation uses [x,y,z,w] but colmap uses [w,x,y,z], 
        # so we convert the quaternion vector.
        quat_ref = [a for a in ref_pose[1:4]]
        quat_ref.append(ref_pose[0])
        rot_ref = Rotation(quat_ref)
        
    # compute the pose of the given image related to the reference image
        delta_rot = rot_ref.__mul__(rot_img.inv())
        rotation = Rotation.from_euler('xyz',tocolmap.dot(delta_rot.as_euler('xyz'))).as_quat()
        translation = tocolmap.dot( rot_ref.inv().as_dcm().dot(trans_ref - trans_img) ) 
              
    except:
        raise ValueError('Input ERROR! Can not parse the given filename to poses')
    
    # convert the quaternion vector to the colmap format
    quat_final = [a for a in rotation[0:3]]
    quat_final.insert(0, rotation[3])
    
    return quat_final, translation


def project_colmap_depth(path_to_file: list=None, output_resolution: tuple=None, 
                               use_radial_dist: bool = False, camera_para: list=None, 
                               save: bool = True) -> np.array:
    '''
        It loads 6 or 4 cubemaps from the given path and merge them to a omnidirectional depth map.
        
        Parameters
        ----------    
        path_to_file: list
            A list of path to the 6 or 4 cubic depthmaps.
            
        output_resolution: tuple
            The resolution of the output omnidirectional image.
            
        use_radial_dist: bool
            Whether to convert the depth to radial distance or not
            
        camera_para: list
            A list of camera parameters: [fx, fy, cx, cy]
            
        save: bool
            Whether to save the result or not.
        
        Return:
        --------
        omnimage: np.array
            The omnidirectional depth map.
        
        Example:
        --------
        >>> estimated = merge_cubedepth_fromcolmap(path_to_file = path_to_dmap,
                                                   output_resolution = [512,1024], 
                                                   use_radial_dist = True, 
                                                   camera_para = [267,267,256,256],
                                                   save = False)
    '''
    if len(path_to_file) != 4 and len(path_to_file) != 6:
        raise ValueError("Input ERROR! Only 4 or 6 depth maps are supported!")
        
    cubemap = CubicMaps()
    cubemap.load_depthmap(path_to_file=path_to_file)
    
    if use_radial_dist:
        if camera_para is None:
            raise ValueError("ERROR during merging cubic depthmaps. Radial distances are required but camera parameters are not given.")
        elif len(camera_para) != 4:
            raise ValueError("Inpute ERROR. Camera parameters should have 4 parameters:[fx, fy, cx, cy].")
        else:
            for ind in range(len(path_to_file)):
                cubemap.depthmap[ind] = cubemap.depth_trans(cubemap.depthmap[ind], camera_para)
    
    cubemap.cube2sphere_fast(resolution = output_resolution)
    
    if save:
        cubemap.save_omnimage()
    
    return cubemap.omnimage
    
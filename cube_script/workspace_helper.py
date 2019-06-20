#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  19 18:39:57 2019

@author: zhantao
"""
import os
import sys
import glob
import warnings
import subprocess
from shutil import copyfile, rmtree
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam360 import Cam360
from spherelib import eu2pol
from cubicmaps import CubicMaps
from read_model import read_cameras_text
from view_selection import view_selection
from scipy.spatial.transform import Rotation

'''
The below shows the transformations between the world coordinate used in Cam360
and the local camera coordinates used by colmap as well as the 6 cubemaps.

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


                                        [world coordinate]
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

BEST_OF_N_VIEWS = 10


def dense_from_cam360list(cam360_list: list, workspace: str, patchmatch_path: str, reference_view: int,
                          views_for_synthesis: int=4, use_view_selection: bool=False, gpu_index: int=-1):
    """
        Given a list of cam360 objects, it calls 'estimate_dense_depth' to estimate 
        depth for all cam360 objects in the list.
        
        Parameters
        ----------    
        cam360_list : list of cam360 objs
            A list containing cam360 objects;
            
        workspace : str
            Where to save the whole work space;
            
        patchmatch_path: str
            Where to find the executable patch matching stereo GPU file. 
            
        reference_view: int
            The index of the reference view. Only works when view selection is disabled;
            
        views_for_synthesis: int
            The number of views (4 or 6) to synthesis the omnidirectional depthmap. 
            4 means the sky and ground will be neglected.      
        
        use_view_selection: bool
            Enable view selection.
            
        gpu_index: int
            The index of GPU to run the Patch Matching.
    """
    if use_view_selection:
        # if enable view selection, reconstruct view by view
        for cnt in range(len(cam360_list)):
            cam360_list = estimate_dense_depth(cam360_list, 
                                               reference_image = cnt,
                                               workspace = workspace,
                                               patchmatch_path = patchmatch_path, 
                                               views_for_synthesis = views_for_synthesis,
                                               use_view_selection = True,
                                               gpu_index = gpu_index)
    else:
        # if disabled view selection, reconstruct all views together
        cam360_list = estimate_dense_depth(cam360_list, 
                                           reference_image = reference_view,
                                           workspace = workspace,
                                           patchmatch_path = patchmatch_path, 
                                           views_for_synthesis = views_for_synthesis,
                                           use_view_selection = False,
                                           gpu_index = gpu_index)    
    return cam360_list


def estimate_dense_depth(cam360_list: list, reference_image: int, workspace: str, patchmatch_path: str, 
                          views_for_synthesis: int=4, use_view_selection: bool=False, gpu_index: int=-1):
    """
        Given a list of cam360 objects, it estimates depthmap for the reference image.
    
        Firstly, it generates cubic maps for all objects. Then it collects cubic maps 
        according to the view (back/front/laft/right) and saves diffetent views to
        corresponding folders (workspace/cubemaps/view0, workspace/cubemaps/view1, etc.).
        At the same time, it generates the camera models as well as the camera poses and
        save these data as .txt file to workspace/cubemaps/parameters/view* for patch 
        matching stereo.
        
        After preparation, it calls Patch Matching Stereo GPU (from colmap) to work on cubic maps and 
        reorganize the estimated depth maps to /workspace/omni_depthmaps/image_name/.
        
        Finally, it reproject the cubic depth maps back to the 360 camera to obtain the 
        omnidirectional depth map.
        
        Parameters
        ----------    
        cam360_list : list of cam360 objs
            A list containing cam360 objects;
            
        reference_image : int
            Index of the reference cam360 object whose local coordinate will be used as 
            the world corrdinate in patch matching stereo;
            
        workspace : str
            Where to save the whole work space;
            
        patchmatch_path: str
            Where to find the executable patch matching stereo GPU file. 
            
        views_for_synthesis: int
            The number of views (4 or 6) to synthesis the omnidirectional depthmap. 
            4 means the sky and ground will be neglected.
            
        use_view_selection: bool
            Enable view selection.
            
        gpu_index: int
            The index of GPU to run the Patch Matching.
        
        Examples
        --------
        >>> estimate_dense_depth(cam360_list = [cam360_1, cam360_2, cam360_3], 
                                  workspace = './workspace',
                                  reference_image = 4,  
                                  patchmatch_path = './colmap', 
                                  views_for_synthesis = 4)
    """
    # create a workspace for patch matching stereo GPU
    scores_list = create_workspace_from_cam360_list(cam_list=cam360_list, refimage_index=reference_image, number_of_views = views_for_synthesis,
                                                    work_dir = workspace, view_selection=use_view_selection)
        
    # run patch matching stereo on each cube views
    print("\n\nExecuting patch match stereo GPU")
    for view in range(views_for_synthesis):
        
        input_path = os.path.join(workspace, "cubemaps/parameters/view" + str(view))
        output_path= os.path.join(workspace, "patch_match_ws/view" + str(view))
        image_path = os.path.join(workspace, "cubemaps/view" + str(view))
        
        check_path_exist(output_path)
        
        command = patchmatch_path + \
                  " image_undistorter" + \
                  " --image_path="  + image_path + \
                  " --input_path="  + input_path + \
                  " --output_path=" + output_path
        CM = subprocess.Popen(command, shell=True)
        CM.wait()
        
        # modify the patch-match.cfg file to set number of source image or 
        # specify the images to be used
        set_patchmatch_cfg(output_path, reference_image, scores_list, view, use_view_selection)
        
        command = patchmatch_path + \
                  " patch_match_stereo" + \
                  " --workspace_path="  + output_path + \
                  " --PatchMatchStereo.depth_min=0"  + \
                  " --PatchMatchStereo.depth_max=500" + \
                  " --PatchMatchStereo.gpu_index={:d}".format(gpu_index) 
        CM = subprocess.Popen(command, shell=True)
        CM.wait()

    # collect cubemaps belonging to same omnidirectional images
    print("\n\nReorganizing workspace ...")
    organize_workspace(workspace=workspace)

    # project cubic depth to omnidirectional depthmap
    print("\n\nReprojecting cubic depth to 360 depth ...")
    resolution = [cam360_list[0]._height, cam360_list[0]._width]
    depth_list = reconstruct_omni_maps(omni_workspace=os.path.join(workspace, 'omni_depthmap/depth_maps/*'), 
                                       view_to_syn=views_for_synthesis, 
                                       maps_type='depth_maps',
                                       resolution=resolution)
    
    # project cost maps
    print("\n\nReprojecting cost maps ...")
    cost_list = reconstruct_omni_maps(omni_workspace=os.path.join(workspace, 'omni_depthmap/cost_maps/*'), 
                                       view_to_syn=views_for_synthesis, 
                                       maps_type='cost_maps',
                                       resolution=resolution)
    
    if use_view_selection:
        # save costs and depth of the reference image to the corresponding object
        cam360_list[reference_image].depth = depth_list[reference_image][:,:,0]
        cam360_list[reference_image].cost = cost_list[reference_image][:,:,0]
        
        rmtree(output_path)     # clean workspace
    else:
        # save all views at a time
        for ind, cam in enumerate(cam360_list):
            cam.depth = depth_list[ind][:,:,0]
            cam.cost = cost_list[ind][:,:,0]
       
    return cam360_list


def set_patchmatch_cfg(workspace: str, reference_image: int, score_list: list, 
                       view_ind : int, enable_view_selection: bool):
    '''
        It keeps the top BEST_OF_N_VIEWS views to reconstruct scenes according
        to the given scores. To reduce computation cost, it only reconstruct 
        the reference view.
        
        Parameters
        ----------            
        workspace : str
            Where to save the whole work space;
        
        reference_image : int
            Index of the reference cam360 object whose local coordinate will be used as 
            the world corrdinate in patch matching stereo;
            
        score_list : list
            A list of scores corresponding to all views. For invalid views, the scores
            are None.
            
        view_ind : int
            The index of the current cubic view. Smaller or equal to views_for_synthesis. 
    '''
    path_to_images = os.path.join(workspace, 'images/*')
    path_to_config = os.path.join(workspace, 'stereo/patch-match.cfg')
    
    # obtain all image names (except for the reference image) and 
    # create a dictionary for images and corresponding valid scores
    src_image = [ image.split('/')[-1] for image in sorted(glob.glob(path_to_images)) if "_{:d}_".format(reference_image+1) not in image ]
    
    if enable_view_selection:
        valid_scores = [score[view_ind] for score in score_list if score[view_ind] is not None]     # load valid scores
    else:
        valid_scores = [0]*len(src_image)       # set scores for images 

    assert len(src_image) == len(valid_scores), "The number of valid images does not match the number of scores"
    image_score_dict = dict(zip(src_image, valid_scores))
    
    # keep the top N views
    top_n_candidates = sorted(image_score_dict.items(), key=lambda item:item[1], reverse=True)[:BEST_OF_N_VIEWS]
    
    # Generate configuration file, e.g.
    #   ref_image
    #   src_image1, src_image2, ... ,src_imageN
    #   src_image1
    #   src_image2, src_image3, ... ,src_imageN, ref_image
    #   ...
    #   src_imageN
    #   ref_image, src_image1, src_image2, ... ,src_imageN-1
    with open(path_to_config, 'r') as file:
        config = file.read()
    
    reference = "cam360_{:d}_view{:d}.png".format(reference_image+1, view_ind)
    source = [ image[0] for image in top_n_candidates]
    config = generate_config(ref=reference, src=source)
    
    with open(path_to_config, 'w') as file:
        file.write(config)
        

def generate_config(ref: str, src: list):
    '''
        It generates configuration contents.
    '''
    config = ""
    for cnt in range(len(src) + 1):
        
        new_task = ref + "\n"
        src_img = [ image +', ' for image in src]
        config = config + new_task + "".join(src_img)[:-2] + "\n"
        
        src.append(ref)
        ref = src[0]
        src = src[1:]
        
    return config
    
def check_path_exist(path: str):
    '''
        If the given path does not exist, it will create the path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def show_processbar(num: int, all_steps: int, text: str):
    '''
        It draws a process bar.
    '''
    sys.stdout.write('\r')
    sys.stdout.write(text)
    sys.stdout.write("[%-20s] %d%%" % ('='*round(num*20/all_steps), 100*(num)/all_steps))
    sys.stdout.flush()


def create_workspace_from_cam360_list( cam_list: list, refimage_index: int = -1, number_of_views: int = 6,
                                       cubemap_resolution: tuple = None, work_dir: str = './workspace', view_selection: bool = False):
    """
        Given a list of cam360 objects, it decomposes these cam360 objs and creates a 
        workspace for patch match stereo GPU.
        
        Firstly, it generates cubic maps for all objects. Then it collects cubic maps 
        according to the view (back/front/laft/right) and saves diffetent views to
        corresponding folders (workspace/cubemaps/view0, workspace/cubemaps/view1, etc.).
        At the same time, it generates the camera models as well as the camera poses and
        save these data as .txt file to workspace/cubemaps/parameters/view* for patch 
        matching stereo.

        Parameters
        ----------    
        cam_list : list of cam360 objs
            A list containing cam360 objects;
            
        refimage_index : int
            Index of the reference cam360 object whose local coordinate will be used as 
            the world corrdinate in patch matching stereo; start from 0;
        
        number_of_views: int 
            Number of views to be decomposed;
        
        cubemap_resolution: tuple
            The resolution of cubic maps.
        
        work_dir : str
            Where to save the whole work space;
            
        view_selection : bool
            whether to select views for dense reconstruction;
        
        Examples
        --------
        >>> create_workspace_from_cam360_list(cam_list=[cam360_1, cam360_2, cam360_3], 
                                              refimage_index=4, 
                                              work_dir = './workspace')
    """
    # verify inputs
    if len(cam_list) < 2:
        raise ValueError("Image is not enough to reconstruct depthmap")
    if refimage_index < 0:
        raise ValueError("Invalid index to reference image")    
    if number_of_views != 4 and number_of_views != 6:
        raise ValueError("Only 4 and 6 views are supported")    
    
    ref_cam = cam_list[refimage_index]   
    score_cam = []
    
    camera_txt_flag = True # whether to writh camera model .txt
    for ind, src_cam in enumerate(cam_list):
        # ckeck the textures
        if src_cam.texture is None:
            if ind == refimage_index:
                raise ValueError("The reference camera360 doesn't have texture")
            else:   
                warnings.warn("The {:d}th camera360 doesn't have valid textures; it will be skipped".format(ind))
                continue            
        else:
            enable_view_selection = (ind!=refimage_index) and view_selection
            
            # decompose omnidirectional image into 6 cubic maps and save them
            scores, camera_txt_flag = decompose_and_save(src_cam, ref_cam, cubemap_resolution, work_dir, prefix="cam360", number_of_views=number_of_views,
                                                             image_index = ind + 1, camera_txt_flag=camera_txt_flag, select_view=enable_view_selection)
            score_cam.append(scores)
                
            # present the process
            show_processbar(ind+1, len(cam_list), text='Creating workspace: ')
            
    return score_cam


def decompose_and_save(cam: Cam360, ref_cam: Cam360, resolution: tuple=None, work_dir: str="./workspace", number_of_views: int=6,
                        prefix: str="cam360_cubemap",  image_index:int=0,  camera_txt_flag: bool=True, select_view: bool=False):
    """
        Given a cam360 objects, it decomposes the cam360 objs into 6 cubic maps
        and save them according to the view (back -> view0, front - view1 etc.).
        At the same time, it generates camera models as well as camera poses and
        save these data as .txt file to workspace/cubemaps/parameters/view*. 
        
        Parameters
        ----------    
        cam : cam360 objs
            A source cam360 objects;
            
        ref_cam : cam360 objs
            The reference cam360 object;
        
        resolution: tuple
            The resolution of cubic maps.
        
        work_dir : str
            Where to save the whole work space;
            
        select_view : bool
            whether to select views for dense reconstruction;
        
        Examples
        --------
        >>> create_workspace_from_cam360_list(cam_list=[cam360_1, cam360_2, cam360_3], 
                                              refimage_index=4, 
                                              work_dir = './workspace')
    """    
    if resolution is None:
        resolution = (cam._height, cam._height)
    
    if cam.texture is None:
        raise ValueError("The given camera360 object doesn't have texture")
    
    else:
        
        cubemap_obj = CubicMaps()
        
        try:  # get the pose of the reference image
            reference_pose = [ref_cam.rotation_mtx, ref_cam.translation_vec]
            reference_cube = cubemap_obj.sphere2cube(ref_cam, resolution=resolution)
        except:
            raise ValueError("The reference camera360 doesn't have valid pose") 
        
        cubemap_obj.sphere2cube(cam, resolution=resolution)
        
        prefix = prefix + "_{:d}".format(image_index)
        new_angle = None
        score_list = []
        for ind in range(number_of_views):
                       
            view_folder = os.path.join(work_dir, 'cubemaps/view' + str(ind))
            para_folder = os.path.join(work_dir, 'cubemaps/parameters/view' + str(ind))
            check_path_exist(view_folder)
            check_path_exist(para_folder)
            
            if select_view:
                # compute initial pose
                intial_z = cam.rotation_mtx.transpose().dot(reference_pose[0].dot( VIEW_ROT[ind].transpose() )).dot(np.array([0,0,1]))
                intial_z = np.expand_dims(intial_z, axis=1)
                initial_pose = eu2pol(intial_z[0], intial_z[1], intial_z[2])
                # select view
                cubemap_obj.cubemap[ind], new_angle, score = view_selection(cam, reference_cube[ind], 
                                                                            initial_pose=(np.abs(initial_pose[1]), np.abs(initial_pose[0])))
            else:
                score = None
            
            score_list.append(score)
            
            if (not select_view) or (score is not None):
            # if no need to select view or the score of the selected view is not None
                # save the view
                image_path = cubemap_obj.save_cubemap(path = view_folder, prefix = prefix, index=[ind])   
                image_name = image_path[0].split('/')[-1]
            
                # TODO: support images taken by different camera models
                
                # save parameters
                camera_parameters = [resolution[0]/2, resolution[1]/2, resolution[0]/2, resolution[1]/2]
                if camera_txt_flag or ( not os.path.exists(os.path.join(para_folder, 'cameras.txt')) ):
                # if there is no parameter file or 
                    camera_id = create_camera_model(path=para_folder, camera_para = [camera_parameters], camera_size=resolution)
                    save_3d_points(para_folder)
                    
                    # only write once
                    camera_txt_flag = False
                else:
                    camera_id = 1
                
                # save poses
                source_pose = [cam.rotation_mtx, cam.translation_vec]
                pose_colmap = convert_coordinate(source_pose, reference_pose, index_of_cubemap=ind, new_angle = new_angle)
                save_pose(para_folder, pose_colmap, image_index, image_name, camera_id)  
        
    return score_list, camera_txt_flag


def convert_coordinate(source_pose: list, reference_pose: list, index_of_cubemap: int, new_angle: tuple=None):
    '''
        It computes the rotation and translation from the reference cube map to 
        the source cube map.
        
        Parameters
        ----------    
        source_pose : list
            The pose of the souorce image. camera -> world
            
        reference_pose : list
            The pose of the reference image. camera -> world
        
        index_of_cubemap : int
            Which cube map is uesd.
        
        new_angle: tuple
            [phi, theta], the angle of the new view
        
    '''
    
    ref_tocolmap = VIEW_ROT[index_of_cubemap,:,:]
    src_tocolmap = VIEW_ROT[index_of_cubemap,:,:]
    
    rotation_ref = reference_pose[0]
    translation_ref = reference_pose[1]
    rotation_source = source_pose[0]
    translation_source = source_pose[1]
   
    if new_angle is not None:
        # mediate view    
        if new_angle[0] == 0:
            med_view2colmap = Rotation.from_euler('x', new_angle[1]).as_euler('zyx')[0]
        elif new_angle[1] == 0:
            med_view2colmap = Rotation.from_euler('z', new_angle[0]).as_euler('zyx')[0]
        else:
            med_view2colmap = Rotation.from_euler('zx', [new_angle[0], new_angle[1]]).as_euler('zyx')
            
        new_view2colmap = med_view2colmap + np.array([np.pi, 0, 0])
        src_tocolmap = Rotation.from_euler('zyx', new_view2colmap).as_dcm()
    
        
    # convert the poses of images from camera->world (in general) to reference->source (colmap)
    rotation_ref2world = rotation_ref.dot( ref_tocolmap.transpose() )
    rotation_world2src = src_tocolmap.dot( rotation_source.transpose() )
    
    delta_rot = rotation_world2src.dot(rotation_ref2world)
    quat_rot  = Rotation.from_dcm(delta_rot).as_quat() 
    quat_rot_colmap = [tmp for tmp in quat_rot[0:3]]
    quat_rot_colmap.insert(0, quat_rot[3])
   
    translation = rotation_world2src.dot(translation_ref - translation_source) 
    
    return [quat_rot_colmap, translation]


def save_pose(para_folder: str, poses: list, image_index: int, image_name: str, camera_id: int):
    '''
        It saves the given poses to the given folder.
        
        Parameters
        ----------    
        para_folder: str
            Where to save the parameters
         
        poses: list
            [[rotation matrix], [translation vector]]
        
        image_index: int 
            The index of the image which has the given poses
            
        image_name: str
            The name of the image which has the given poses
            
        camera_id: int
            The index of the camera model coresponding to the image 
    '''
    
    # record image id, pose, camera id as well the image name to the images.txt in colmap format
    with open(os.path.join(para_folder, 'images.txt'), "a+") as f:
        f.write(str(image_index))
        for quat in poses[0]:
            f.write( ' ' + str(quat))
        for trans in poses[1]:
            f.write( ' ' + str(trans))
        f.write( ' ' + str(camera_id) + ' ' + image_name)
        f.write('\n\n')
        f.close()    


def save_3d_points(path: str):
    '''
        It creates an empty point3D.txt file at the given path.
    '''
    with open(os.path.join(path, 'points3D.txt'), "a+") as f: 
        f.closed
        

def create_depth_workspace(image_dir: str='', file_suffix: str='exr', work_dir: str='./',
                           radius_depth: Optional[bool] = False, cam_para: Optional[list] = None,
                           resolution: tuple = (256,256)):
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
            
        resolution: tuple
            The resolution of cube maps.

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
        cubemap_obj.sphere2cube(Omni_obj, resolution=resolution, is_depth = True)
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
            1 PINHOLE 512 512 358.29301602 358.29301602 256 256 \n
            2 PINHOLE 512 512 267.29301602 267.29301602 256 256
            
        Return the number of existing cameras models
        
        Examples
        --------
        >>> create_camera_model(path='./', camera_para = [267,267,256,256], image_size=(512,512))        
    """
    if camera_para is None or camera_size is None:
        raise ValueError("Input ERROR! Camera parameters or image size are not provided") 
    elif len(camera_size) != 2:
        raise ValueError("Input ERROR! Invalid image size")
    else:
        camera_file = os.path.join(path, 'cameras.txt')
        
        helper = '# Camera list with one line of data per camera: \n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n# Number of cameras: 1 \n' 
        
        try:
            previous_camera = len(read_cameras_text(camera_file))
        except:
            previous_camera = 0
        
        with open( camera_file, "a+") as f:   # Opens file and casts as f 
            f.write(helper)
            for camera_id in range(len(camera_para)):
                # Writing
                f.write(str(camera_id + 1 + previous_camera) + ' PINHOLE ' + str(camera_size[0]) + ' ' + str(camera_size[1]))
                for para in camera_para[camera_id]:
                    f.write( ' ' + str(para))
                f.write('\n')
                f.close()         # Close file

    return camera_id + 1 + previous_camera


def organize_workspace(workspace: str):
    '''
        For each omnidirectional image, it copies the corresponding cubic depth maps
        to: /workspace/omni_depthmap/image_name
        
        Parameters
        ----------    
        workspace: str
            The path to the workspace.
    '''
    depth_path = os.path.join(workspace, 'patch_match_ws')
    dst_folder = os.path.join(workspace, 'omni_depthmap')
    
    check_path_exist(dst_folder)
    
    # reorganize depth maps and cost maps
    organize_outputs(colmap_ws=depth_path, dst_folder=dst_folder, target = 'depth_maps')
    organize_outputs(colmap_ws=depth_path, dst_folder=dst_folder, target = 'cost_maps')
    
    
def organize_outputs(colmap_ws: str, dst_folder: str, target: str):
    '''
        It collect all target files under the colmap_ws to dst_folder according to
        filenames.
        
        colmap_ws: str
            path to the colmap workspace
        
        dst_folder: str
            destination folder
            
        target: str
            file to be reorganized, e.g. depth_map, cost_map and normal_map
    '''
    for image in sorted(glob.glob( os.path.join(colmap_ws, 'view0/stereo/' + target + '/*.geometric.bin' ))):
        
        image_name = image.split('/')[-1].split('.')[0][:-6]
        
        for depth_view in sorted(glob.glob( os.path.join(colmap_ws, 'view*') )):
            view_num = depth_view.split('/')[-1]
            depth_file = glob.glob( os.path.join(colmap_ws, view_num+'/stereo/' + target + '/{:s}*.geometric.bin'.format(image_name)) )[0]
            
            dst_path = os.path.join(dst_folder, target + '/{:s}'.format(image_name)) 
            check_path_exist(dst_path)
            
            copyfile(depth_file, os.path.join(dst_path, '{:s}_{:s}.geometric.bin'.format( image_name, view_num))) 


def reconstruct_omni_maps(omni_workspace: str, Camera_parameter: list = None, view_to_syn: int = 6, 
                          resolution: list = None, maps_type: str = "depth_maps", save_omni: bool = True):
    '''
        It projects depth maps to omnidirectional depth map.
        
        Parameters
        ----------    
        omni_workspace: str
            The path to the /workspace/omni_depthmap
        
        Camera_parameter: list
            Camera paraeters to be used for projection
            
        view_to_syn: int
            The number of views (4 or 6) to synthesis the omnidirectional depthmap. 
            4 means the sky and ground will be neglected.
            
        resolution: list
            The resolution of ouput depth map.
            
        maps_type: str
            Type of the maps to be projected, options: [depth_maps, cost_maps, normal_maps]
            
        save_omni: bool
            Whether to save the omnidirectional depth map.
            
    '''
    depth_list = []
    for folder in sorted( glob.glob( omni_workspace )):
        file_name = folder.split('/')[-1]
        omni_depth = project_colmap_maps(path=folder,
                                          view_name=file_name,
                                          views_list=[num for num in range(view_to_syn)],
                                          output_resolution=resolution, 
                                          use_radial_dist=True, 
                                          camera_para=Camera_parameter,
                                          map_type=maps_type,
                                          save=save_omni)
        depth_list.append(omni_depth)
    return depth_list
        

def project_colmap_maps(path: str, view_name: str = None, views_list: list = [],
                         output_resolution: tuple=None, use_radial_dist: bool = False,
                         camera_para: list=None, map_type: str="depth_maps", save: bool = True) -> np.array:
    '''
        It loads 6 or 4 cubemaps from the given path and merge them to a omnidirectional depth map.
        
        Parameters
        ----------    
        path: str
            The path to the 6 views. For example, '../depthmap/castle/fixed/allviews/'
            
        view_name: str
            The name of the view. For example, 'test_0_0_1_1024_5120000';
            
        views_list: list
            A list of integer denoting the views to be loaded;
            
        output_resolution: tuple
            The resolution of the output omnidirectional image.
            
        use_radial_dist: bool
            Whether to convert the depth to radial distance or not
            
        camera_para: list
            A list of camera parameters: [fx, fy, cx, cy]
        
        map_type: str
            Type of the maps to be projected, options: [depth_maps, cost_maps, normal_maps]
            
        save: bool
            Whether to save the result or not.
        
        Return:
        --------
        omnimage: np.array
            The omnidirectional depth map.
        
        Example:
        --------
        >>> estimated = project_colmap_maps(path = path_to_dmap, view_name = name_pattern,
                                 views_list = [0,1,2,3], output_resolution=(512, 1024), 
                                 use_radial_dist=True,  camera_para=[256,256,256,256], save=False)
    '''
    if len(views_list) <= 0:
        raise ValueError("Input ERROR! Please specify the views to be loaded.")
    elif view_name is None:
        raise ValueError("Input ERROR! Please specify the name of the views to be loaded.")
    else:
        path_to_file = []
        for ind in views_list:
            path_to_file.append( os.path.join(path, view_name + '_view' + str(ind) + '.*'))
    
    cubemap = CubicMaps()
    cubemap.load_depthmap(path_to_file=path_to_file, type_ = map_type)
    
    if use_radial_dist:
        if camera_para is None:
            camera_para = [ cubemap.depthmap[0].shape[0]/2 ] * 4 
        elif len(camera_para) != 4:
            raise ValueError("Inpute ERROR. Camera parameters should have 4 parameters:[fx, fy, cx, cy].")
        
        for ind in range(len(path_to_file)):
            if map_type == 'depth_maps':
                cubemap.depthmap[ind] = cubemap.depth_trans(cubemap.depthmap[ind], camera_para)
            elif map_type == 'normal_maps':
                warnings.warn("Project normal maps are not supported by now.")
    
    cubemap.cube2sphere_fast(resolution = output_resolution)
    
    if save:
        if map_type == 'depth_maps' or 'cost_maps':
            cubemap.save_omnimage(path=path, name='360_'  + map_type + '.exr')
        else:
            cubemap.save_omnimage(path=path, name='360_'  + map_type + '.png')
    
    return cubemap.omnimage
    

def evaluate(estimation: np.array, GT: np.array, checking_line: int = 100, save: bool = False):   
    '''
        It compares the two given images. 
        
        Parameters
        ----------    
        estimation: np.array
            Estimated depth map.
            
        GT: np.array
            The ground truth.
            
        checking_line: int
            The position of the vertical line along which the depth will be ploted.
            
        camera_para: list
            A list of camera parameters: [fx, fy, cx, cy]
            
        save: bool
            Whether to save the estimated depth map, ground truth as well as the difference map.
        
        Return:
        --------
            Raw RMSE and RMSE, the raw means that the depth errors from all 
            pixels are counted, including the sky and the ground.
        
        Example:
        --------
        >>> estimated = merge_cubedepth_fromcolmap(path_to_file = path_to_dmap,
                                                   output_resolution = [512,1024], 
                                                   use_radial_dist = True, 
                                                   camera_para = [267,267,256,256],
                                                   save = False)
    '''
    
    plt.figure(figsize=[12,12])
    min_d_toshow = min(estimation.min(), GT.min())
    max_d_toshow = max(estimation.max(), GT.max())

    if not save:
        plt.subplot(321)
        plt.imshow(estimation, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow);
        plt.axis('off')
        plt.title('Estimated Depth')
        
        plt.subplot(322)
        plt.imshow(GT, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
        plt.axis('off')
        plt.title('Ground Truth')
        
        plt.subplot(323)
        plt.plot(GT[:,checking_line]);
        plt.plot(estimation[:,checking_line]);
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the vertical line at ' + str(checking_line) + 'th column')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
    
        plt.subplot(324)
        errors = abs(GT[:,checking_line] - estimation[:,checking_line])
        errors[errors > 20] = 0
        plt.plot(errors)
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the vertical line at ' + str(checking_line) + 'th column')
        plt.grid()
        
        plt.subplot(325)
        plt.plot(GT[checking_line,:]);
        plt.plot(estimation[checking_line,:]);
        plt.xlabel('left ------------------> right \n ')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the horizontal line at ' + str(checking_line) + 'th row')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
    
        plt.subplot(326)
        errors = abs(GT[checking_line,:] - estimation[checking_line,:])
        errors[errors > 20] = 0
        plt.plot(errors)
        plt.xlabel('left ------------------> right')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the horizontal line at ' + str(checking_line) + 'th row')
        plt.grid()
        plt.tight_layout()
    else:   
        plt.imshow(estimation, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow);
        plt.axis('off')
        plt.savefig('estimated_depthmap.png', dpi=300, bbox_inches="tight")
        
        plt.imshow(GT, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
        plt.axis('off')
        plt.savefig('Groundtruth_depthmap.png', dpi=300, bbox_inches="tight")
    
        plt.figure()
        plt.plot(GT[:,checking_line]);
        plt.plot(estimation[:,checking_line]);
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the vertical line at ' + str(checking_line) + 'th column')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
        plt.savefig('Depth_along' + str(checking_line) + 'th_column.png', dpi = 300)
        
        plt.figure()
        errors = abs(GT[:,checking_line] - estimation[:,checking_line])
        errors[errors > 20] = 0
        plt.plot(errors)
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the vertical line at ' + str(checking_line) + 'th column')
        plt.grid()
        plt.savefig('Absolute_error_along' + str(checking_line) + 'th_column.png', dpi = 300)
        
        plt.figure()
        plt.plot(GT[checking_line,:]);
        plt.plot(estimation[checking_line,:]);
        plt.xlabel('left ------------------> right \n ')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the horizontal line at ' + str(checking_line) + 'th row')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
        plt.savefig('Depth_along' + str(checking_line) + 'th_row.png', dpi = 300)
        
        plt.figure()
        errors = abs(GT[checking_line,:] - estimation[checking_line,:])
        errors[errors > 20] = 0
        plt.plot(errors)
        plt.xlabel('left ------------------> right')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the horizontal line at ' + str(checking_line) + 'th row')
        plt.grid()
        plt.savefig('Absolute_error_along' + str(checking_line) + 'th_row.png', dpi = 300)
    
    # calculate rmse from all pixels
    diff_map = abs(estimation - GT)
    raw_RMSE = np.sqrt(np.sum(diff_map**2)/(GT.size))
    print('The raw RMSE is:', raw_RMSE )

    # calculate rmse from estimated pixels
    mask = np.logical_and(estimation!=0, diff_map<20)
    masked_RMSE = np.sqrt(np.sum(diff_map[mask]**2)/(mask.size))
    print('The masked RMSE is:', masked_RMSE )

    # to make the error map clear, set this threshold
    threshold = 10
    diff_map[np.abs(diff_map)>threshold] = threshold

    plt.figure(figsize=[10,6])
    plt.imshow(np.abs(diff_map), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=threshold)
    plt.colorbar()
    plt.title('Absolute Error map -- Green(small error) to red(large error)')
    if save:
        plt.savefig('Error_maps.png', dpi=300, bbox_inches="tight")

    return raw_RMSE


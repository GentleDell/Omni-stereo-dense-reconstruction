#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  19 18:39:57 2019

@author: zhantao
"""
import os
import glob
import warnings
import subprocess
from shutil import rmtree
from shutil import copyfile
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam360 import Cam360
from cubicmaps import CubicMaps
from spherelib import eu2pol, pol2eu
from view_synthesis import synthesize_view
from read_model import read_cameras_text
from view_selection import view_selection, sparseMatches as Matches, SAVE_REFERANCE, SAVE_SOURCE
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

# FUNCTION SWITCHES -- if enable initialization, the reference camera must be the first element of the cam360 list
DISABLE_INIT = True

# small number
EPS = 1e-8

# rotation matrixs from cam360 coordinate to the 6 views (cubmaps)
VIEW_ROT = np.array([[[-1,0,0], [0,0,-1], [0,-1,0]],
                     [[ 0,1,0], [0,0,-1], [-1,0,0]],
                     [[ 1,0,0], [0,0,-1], [0, 1,0]],
                     [[0,-1,0], [0,0,-1], [1, 0,0]],
                     [[ 1,0,0], [0, 1,0], [0,0, 1]],
                     [[ 1,0,0], [0,-1,0], [0,0,-1]]])

# theta and phi, image coordinate i.e. (-y of the world corrdinate) is y axis;
# (-x of the world coordinate) is x axis while the z axis is the same as the 
# world coordniate. 
VIEW_ANG = np.array([[np.pi/2,    0   ],
                     [np.pi/2, np.pi/2],
                     [np.pi/2, np.pi  ],
                     [np.pi/2, np.pi*3/2],
                     [   0   ,  np.pi ],
                     [ np.pi ,  np.pi]])

# field of view of SRC cubic maps
CUBICVIEW_FOV   = np.pi*2/3

BEST_OF_N_VIEWS = 20


def dense_from_cam360list(cam360_list: list, reference_image: int, workspace: str, 
                          patchmatch_path: str, views_for_depth: int=4, use_view_selection: bool=False, 
                          gpu_index: int=-1, use_geometric_filter: bool=False, seed: float = None, debug_mod: bool=False):
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
            
        views_for_depth: int
            The number of views (4 or 6) to synthesis the omnidirectional depthmap. 
            4 means the sky and ground will be neglected.
            
        use_view_selection: bool
            Enable view selection.
            
        gpu_index: int
            The index of GPU to run the Patch Matching.
            
        use_geometry: bool
            Enable geometric filtering.
        
        Examples
        --------
        >>>> cam360_list = dense_from_cam360list(cam360_list, 
                        workspace = args.workspace,
                        patchmatch_path = args.patchmatch_path, 
                        reference_image  = args.reference_view,
                        views_for_depth = args.views_for_depth,
                        use_view_selection = args.view_selection,
                        gpu_index = args.gpu_index,
                        use_geometry = args.geometric_depth)
    """
    # clean existing workspace
    if os.path.isdir(workspace):
        rmtree(workspace)
    
    # set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # create a workspace for patch matching stereo GPU
    scores_list, name_list = create_workspace_from_cam360_list(cam_list=cam360_list, refimage_index=reference_image, number_of_views = views_for_depth,
                                                    work_dir = workspace, view_selection=use_view_selection)

    # run patch matching stereo on each cube views
    print("\n\nExecuting patch match stereo GPU")
    for view in range(views_for_depth):
        
        # define paths
        input_path = os.path.join(workspace, "cubemaps/parameters/view" + str(view))
        output_path= os.path.join(workspace, "patch_match_ws/view" + str(view))
        image_path = os.path.join(workspace, "cubemaps/view" + str(view))
        
        check_path_exist(output_path)
        
        # undistort images and orgnize workspace for dense reconstruction
        command = patchmatch_path + \
                  " image_undistorter" + \
                  " --image_path="  + image_path + \
                  " --input_path="  + input_path + \
                  " --output_path=" + output_path
        CM = subprocess.Popen(command, shell=True)
        CM.wait()
        
        # modify the patch-match.cfg file to specify the images to be used
        set_patchmatch_cfg(output_path, reference_image, scores_list, name_list, view, use_geometric_filter)
        
        # start patch matching stereo
        command = patchmatch_path + \
                  " patch_match_stereo" + \
                  " --workspace_path="  + output_path + \
                  " --PatchMatchStereo.depth_min=0"  + \
                  " --PatchMatchStereo.depth_max=500" + \
                  " --PatchMatchStereo.gpu_index={:d}".format(gpu_index) 
        if use_geometric_filter:
            command = command + " --PatchMatchStereo.geom_consistency true"     # use geometry filtering
        else:
            command = command + " --PatchMatchStereo.geom_consistency false"    # only compute the photometric depth
            
        CM = subprocess.Popen(command, shell=True)
        CM.wait()
            

    # collect cubemaps belonging to same omnidirectional images
    print("\n\nReorganizing workspace ...")
    organize_workspace(workspace=workspace, is_geometric=use_geometric_filter)

    # project cubic depth to omnidirectional depthmap
    print("\n\nReprojecting cubic depth to 360 depth ...")
    resolution = [cam360_list[0]._height, cam360_list[0]._width]
    depth_list = reconstruct_omni_maps(omni_workspace = os.path.join(workspace, 'omni_depthmap/depth_maps/*'), 
                                       view_to_syn = views_for_depth, 
                                       maps_type   = 'depth_maps',
                                       resolution  = resolution,
                                       enable_geom = use_geometric_filter)
    
    # project cost maps
    print("\n\nReprojecting cost maps ...")
    cost_list = reconstruct_omni_maps(omni_workspace = os.path.join(workspace, 'omni_depthmap/cost_maps/*'), 
                                      view_to_syn = views_for_depth, 
                                      maps_type   = 'cost_maps',
                                      resolution  = resolution,
                                      enable_geom = use_geometric_filter)

    world2ref = cam360_list[reference_image].rotation_mtx
    if use_geometric_filter and not use_view_selection:
        # save all views at a time        
        for ind, cam in enumerate(cam360_list):
            # rotate depth maps for the corresponding source view
            depth = Cam360(rotation_mtx = np.eye(3), translation_vec=np.array([0,0,0]), 
                   height = depth_list[ind].shape[0], width = depth_list[ind].shape[1], channels = depth_list[ind].shape[2], 
                   texture= depth_list[ind][:,:,0]/255)
            
            cost = Cam360(rotation_mtx = np.eye(3), translation_vec=np.array([0,0,0]), 
                   height = cost_list[ind].shape[0], width = cost_list[ind].shape[1], channels = cost_list[ind].shape[2], 
                   texture= cost_list[ind][:,:,0]/255)
        
            cam.depth = depth.rotate( cam.rotation_mtx.dot(world2ref.transpose()) ).texture[:,:,0]*255
            cam.cost  = cost.rotate ( cam.rotation_mtx.dot(world2ref.transpose()) ).texture[:,:,0]*255
    else:                
        cam360_list[reference_image].depth = depth_list[0][:,:,0]
        cam360_list[reference_image].cost  = cost_list [0][:,:,0]
    
    # if intermediate results are not required (not in debug mode), remove them all 
    if not debug_mod:
        rmtree(workspace)
    
    return cam360_list


def set_patchmatch_cfg(workspace: str, reference_image: int, score_list: list, name_list: list,
                       view_ind : int, use_geometry: bool):
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
            The index of the current cubic view. Smaller or equal to views_for_depth. 
        
        use_geometry: bool
            Enable geometric filtering.
        
    '''
    path_to_config = os.path.join(workspace, 'stereo/patch-match.cfg')
    
    # obtain all image names (except for the reference image) and 
    # create a dictionary for images and corresponding valid scores
    
    valid_scores = []
    source_image = []
    for names, scores in zip(name_list, score_list):
        mask = [ True if'view{:d}'.format(view_ind) in name and "_{:d}_".format(reference_image+1) not in name 
                else False 
                for name in names ]
        source_image = source_image + np.array(names)[mask].tolist()
        valid_scores = valid_scores + np.array([score for score in scores if score is not None])[mask].tolist()

    assert len(source_image) == len(valid_scores), "The number of valid images does not match the number of scores"
    image_score_dict = dict(zip(source_image, valid_scores))
    
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
    config = generate_config(ref=reference, src=source, geom = use_geometry)
    
    with open(path_to_config, 'w') as file:
        file.write(config)
        

def generate_config(ref: str, src: list, geom: bool):
    '''
        It generates configuration contents.
    '''
    config = ""
    
    if geom: 
        loop = len(src) + 1    # if use geometric filter, colmap needs photometric depth of all images;
    else:
        loop = 1               # if not and by default we only want the photometric depth for the ref image;
        
    for cnt in range(loop):
        
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
    views_score = []
    views_names = []
    
    selected_view_index = np.zeros(number_of_views)
    camera_txt_flag = True # whether to writh camera model .txt
    for ind, src_cam in enumerate(cam_list):
        # present the process
        print('Decomposing the {:d} image'.format(ind))
        
        # ckeck the textures
        if src_cam.texture is None:
            if ind == refimage_index:
                raise ValueError("The reference camera360 doesn't have texture")
            else:   
                warnings.warn("The {:d}th camera360 doesn't have valid textures; it will be skipped".format(ind))
                continue            
        else:
            enable_view_selection = (ind!=refimage_index) and view_selection  # not run view selection on the reference view itself
            
            # decompose omnidirectional image into 6 cubic maps and save them
            scores, img_names, camera_txt_flag, num_selected_view = decompose_and_save(src_cam, ref_cam, cubemap_resolution, work_dir, prefix="cam360", 
                                                                            number_of_views=number_of_views, image_index = ind+1, refIndex = refimage_index, view_index = selected_view_index,
                                                                            camera_txt_flag = camera_txt_flag, select_view = enable_view_selection)
            views_score.append(scores)  
            views_names.append(img_names)
            
    return views_score, views_names


def decompose_and_save(src_cam: Cam360, ref_cam: Cam360, resolution: tuple, work_dir: str, prefix: str,  image_index: int, 
                       view_index: np.array, refIndex = int, number_of_views: int=6, camera_txt_flag: bool=True, select_view: bool=False):
    """
        Given a cam360 objects, it decomposes the cam360 objs into 6 cubic maps
        and save them according to the view (back -> view0, front - view1 etc.).
        At the same time, it generates camera models as well as camera poses and
        save these data as .txt file to workspace/cubemaps/parameters/view*. 
        
        Parameters
        ----------    
        src_cam : cam360 objs
            A source cam360 objects;
            
        ref_cam : cam360 objs
            The reference cam360 object;
        
        resolution: tuple
            The resolution of cubic maps.
        
        work_dir : str
            Where to save the whole work space;
            
        select_view : bool
            whether to select views for dense reconstruction;
    """    
    if resolution is None:
        resolution = (src_cam._height, src_cam._height)
    
    if src_cam.texture is None:
        raise ValueError("The given camera360 object doesn't have texture")
    
    else:
        
        cubemap_obj = CubicMaps()
        
        try:  # get the pose of the reference image
            reference_pose = [ref_cam.rotation_mtx, ref_cam.translation_vec]
            reference_cube = cubemap_obj.sphere2cube(ref_cam, resolution=resolution, fov=CUBICVIEW_FOV)
            
        except:
            raise ValueError("The reference camera360 doesn't have valid pose") 
        
        cubemap_obj._cubemap = [[]]*number_of_views
        
        # align src and ref camera to reduce complexity
        src2ref = src_cam.rotation_mtx.transpose().dot(reference_pose[0])
        cam = src_cam.rotate(src2ref)
                
        prefix = prefix + "_{:d}".format(image_index)
        score_list = []
        name_list  = []
        for ind in range(number_of_views):
            
            initial_pose=VIEW_ANG[ind]
            
            view_folder = os.path.join(work_dir, 'cubemaps/view' + str(ind))
            para_folder = os.path.join(work_dir, 'cubemaps/parameters/view' + str(ind))
            check_path_exist(view_folder)
            check_path_exist(para_folder)            
            
            score = []
            # If view selection si enabled, the function will 
            # try to find a view with enough matches and triangulation 
            # angle. If it is not found, the cubic map of the view
            # will be used.
            if select_view:
                # horizontal views
                if ind < 4:
                    cubemap_obj.cubemap[ind], initial_pose, score_vs, matches = view_selection(
                                       cam, 
                                       reference_cube[ind], 
                                       initial_pose, 
                                       reference_trans=reference_pose[1],
                                       fov=(CUBICVIEW_FOV, CUBICVIEW_FOV))
                # the top and bottom views need to be rotated so that them can be treated 
                # as horizontal views.
                elif ind == 4:
                    cubemap_obj.cubemap[ind], initial_pose, score_vs, matches = view_selection(
                                       cam.rotate(alpha = (np.pi/2,0,0), order = (0,1,2)) , 
                                       reference_cube[ind], 
                                       initial_pose = VIEW_ANG[2], 
                                       reference_trans = reference_pose[1],
                                       fov=(CUBICVIEW_FOV, CUBICVIEW_FOV))
                elif ind == 5:                    
                    cubemap_obj.cubemap[ind], initial_pose, score_vs, matches = view_selection(
                                       cam.rotate(alpha = (-np.pi/2,0,0), order = (0,1,2)), 
                                       reference_cube[ind], 
                                       initial_pose = VIEW_ANG[2], 
                                       reference_trans = reference_pose[1],
                                       fov=(CUBICVIEW_FOV, CUBICVIEW_FOV))
                else:
                    ValueError("invalid index @ decompose_and_save()")
                    
                viewSelectSuccess = score_vs is not None
                if not viewSelectSuccess:
                    score_vs = [1.5]
                    initial_pose=VIEW_ANG[ind]
                    cubemap_obj.cubemap[ind] = cubemap_obj.cube_projection(cam = cam,
                                   direction  = np.flip(initial_pose).tolist()+[CUBICVIEW_FOV, CUBICVIEW_FOV],  # since in cubeprojection, the angle is (phi, theta)
                                   resolution = resolution)   
                
                # For comparison, should be commented
                if DISABLE_INIT:
                    matches = Matches([],[],[])
                    
                score.append(score_vs) 
                view_index[ind] += 1
                image_name_vs, camera_txt_flag = save_view_parameters(
                        view_folder, para_folder, prefix + '_selected', ind, 
                        int(view_index[ind]), refIndex, camera_txt_flag, 
                        cam, cubemap_obj, resolution, 
                        initial_pose, reference_pose, 
                        viewSelectSuccess, matches)
                
                name_list.append(image_name_vs)
                
            else:
                # regular cubic maps will be generated and saved
                cubemap_obj.cubemap[ind] = cubemap_obj.cube_projection(cam = cam,
                                   direction  = np.flip(initial_pose).tolist()+[CUBICVIEW_FOV, CUBICVIEW_FOV],  # since in cubeprojection, the angle is (phi, theta)
                                   resolution = resolution)
                score = [2]
                view_index[ind] += 1
                matches = Matches([],[],[])
                image_name, camera_txt_flag = save_view_parameters(view_folder, para_folder, prefix, ind, int(view_index[ind]), refIndex, camera_txt_flag, 
                                                       cam, cubemap_obj, resolution, initial_pose, reference_pose, False, matches)
                name_list.append(image_name)
        
            score_list = score_list + score
            
    return score_list, name_list, camera_txt_flag, image_index


def save_view_parameters(target_folder: str, para_folder: str, prefix: str, 
                         view_ind: int, image_index: int, refIndex: int, write_camera: bool, 
                         cam: Cam360, cubemap_obj: CubicMaps, resolution: tuple , 
                         initial_pose: tuple, reference_pose: tuple, 
                         withViewSelection: bool, sparseMatches: Matches):
    
    image_path = cubemap_obj.save_cubemap(path = target_folder, prefix = prefix, index=[view_ind])   
    image_name = image_path[0].split('/')[-1]

    # TODO: support images taken by different camera models
    
    # save parameters
    camera_parameters = [resolution[0]/2/np.tan(CUBICVIEW_FOV/2), resolution[1]/2/np.tan(CUBICVIEW_FOV/2), resolution[0]/2, resolution[1]/2]
    if write_camera or ( not os.path.exists(os.path.join(para_folder, 'cameras.txt')) ):
    # if there is no parameter file or no need to write a file  
        camera_id = create_camera_model(path=para_folder, camera_para = [camera_parameters], camera_size=resolution)
        
        # only write once
        write_camera = False
        
    else:
        camera_id = 1
    
    # save poses
    source_pose = [cam.rotation_mtx, cam.translation_vec]
    pose_colmap = convert_coordinate(
            source_pose, reference_pose, 
            index_of_cubemap=view_ind, 
            initial_pose = initial_pose, 
            withViewSelection=withViewSelection
            )
    save_pose(para_folder, pose_colmap, image_index, image_name, camera_id)  
    
    # save points
    f = open(os.path.join(para_folder, 'points3D.txt'), 'a+')
    f.close()
    if len(sparseMatches.keyPointRef) > 0:       
        cameraPara = np.array([[camera_parameters[0], 0, camera_parameters[2]], 
                               [0, camera_parameters[1], camera_parameters[3]],
                               [0, 0, 1]])
        sparseMatches.setIntrinsics(cameraPara)
        
        quaternion = [tmp for tmp in pose_colmap[0][1:4]] + [pose_colmap[0][0]]
        rotation   = Rotation.from_quat(quaternion).as_dcm()
        sparseMatches.setExtrinsics(rotation, np.array(pose_colmap[1]))
    
        sparseMatches.triangulateMatches()
        sparseMatches.savePoints(para_folder, refIndex + 1, image_index) 
        
    return image_name, write_camera



def convert_coordinate(source_pose: list, reference_pose: list, index_of_cubemap: int, 
                       initial_pose: tuple=None, withViewSelection : bool = False):
    '''
        It computes the rotation and translation from the source view to 
        the reference view.
        
        Parameters
        ----------    
        source_pose : list
            The pose of the souorce image. 
            rotation: world -> camera, translation: camera -> world under camera
            
        reference_pose : list
            The pose of the reference image. 
            rotation: world -> camera, translation: camera -> world under camera
        
        index_of_cubemap : int
            Which cube map is uesd.
        
        initial_pose: tuple
            [theta, phi], the angle of the projected view
        
    '''
    
    ref_tocolmap = VIEW_ROT[index_of_cubemap,:,:]
    src_tocolmap = VIEW_ROT[index_of_cubemap,:,:]
    
    rotation_ref = reference_pose[0]        # world -> camera
    rotation_src = source_pose[0]           # world -> camera
    translation_ref = reference_pose[1]     # camera -> world under camera <=> world -> camera
    translation_src = source_pose[1]        # camera -> world under camera <=> world -> camera
   
    # Horizontal 4 views
    if index_of_cubemap < 4 or not withViewSelection:
        if initial_pose[1] == 0:
            med_view2colmap = Rotation.from_euler('x', initial_pose[0]).as_euler('zyx')
        elif initial_pose[0] == 0:
            med_view2colmap = Rotation.from_euler('z', initial_pose[1]).as_euler('zyx')
        else:
            med_view2colmap = Rotation.from_euler('zx', [initial_pose[1], initial_pose[0]]).as_euler('zyx')

        new_view2colmap = med_view2colmap + np.array([np.pi, 0, 0])    # convert from image coordinate to local coordinate
        src_tocolmap = Rotation.from_euler('zyx', new_view2colmap).as_dcm()
    # Top view 
    elif index_of_cubemap == 4:
        vec_tra = pol2eu(initial_pose[0], initial_pose[1] - np.pi, 1)   # pol2eu() is under local coordinate instead of image coordinate
        vec_ori = np.array([[-1,0,0],[0,0,-1],[0,-1,0]]).dot(vec_tra)
        angleX  = np.sign(vec_ori[1]) * np.arccos( np.sqrt(vec_ori[0]**2 + vec_ori[2]**2) / np.linalg.norm(vec_ori) )
        angleY  = -1*np.sign(vec_ori[0]) * np.arccos( abs(vec_ori[2]) / np.sqrt(vec_ori[0]**2 + vec_ori[2]**2) )
        
        if angleX == 0:
            med_view2colmap = Rotation.from_euler('y', angleY).as_euler('zyx')
        elif angleY == 0:
            med_view2colmap = Rotation.from_euler('x', angleX).as_euler('zyx')
        else:
            med_view2colmap = Rotation.from_euler('yx', [angleY, angleX]).as_euler('zyx')

        new_view2colmap = med_view2colmap
        src_tocolmap = Rotation.from_euler('zyx', new_view2colmap).as_dcm()
    # Bottom view
    else:
        vec_tra = pol2eu(initial_pose[0], initial_pose[1] - np.pi, 1)
        vec_ori = np.array([[-1,0,0],[0,0,1],[0,1,0]]).dot(vec_tra)
        angleX  = -1*np.sign(vec_ori[1]) * np.arccos( np.sqrt(vec_ori[0]**2 + vec_ori[2]**2) / np.linalg.norm(vec_ori) )
        angleY  = np.sign(vec_ori[0]) * np.arccos( abs(vec_ori[2]) / np.sqrt(vec_ori[0]**2 + vec_ori[2]**2) )
        
        if angleX == 0:
            med_view2colmap = Rotation.from_euler('y', angleY).as_euler('zyx')
        elif angleY == 0:
            med_view2colmap = Rotation.from_euler('x', angleX).as_euler('zyx')
        else:
            med_view2colmap = Rotation.from_euler('yx', [angleY, angleX]).as_euler('zyx')

        new_view2colmap = med_view2colmap + np.array([0, 0, np.pi])
        src_tocolmap = Rotation.from_euler('zyx', new_view2colmap).as_dcm()
        
    
    # Here rotations and translations are converted from cam360 to colmap
    # 
    # In cam360, rotation R is from WORLD to CAMERA; translation t is from CAMERA
    # to WORLD (expressed under camera coordinates). [R, t] is the T matrix. So, 
    # given a point Pw from world coordinate, the corresponding coordinate P_cam
    # is P_cam = [R | t] * Pw 
    # 
    # In colmap (image.txt), rotations and translations are from WORLD to CAMERA 
    # (under world coordinate), which is similar to the coordinate used in cam360.
    #
    # During dense reconstructions, coordinates of reference images are world coords.
    # Thus, poses for colmap can be calculated as: (ref = reference omni; src= source omni)
    #       R_refview2srcview = R_src2colmap * R_src * R_ref.inv() * R_ref2colmap.inv()
    #       t_refview2srcview = - R_src2colmap * R_src * R_ref.inv() * t_ref + R_src2colmap * t_src
    
    rotation_refview2world = rotation_ref.transpose().dot( ref_tocolmap.transpose() )
    rotation_world2srcview = src_tocolmap.dot( rotation_src )
    
    ref2src_rot = rotation_world2srcview.dot(rotation_refview2world)
    quat_rot  = Rotation.from_dcm(ref2src_rot).as_quat() 
    quat_rot_colmap = [tmp for tmp in quat_rot[0:3]]
    quat_rot_colmap.insert(0, quat_rot[3])
    
    translation = np.array( [t for t in (-ref2src_rot.dot( ref_tocolmap.dot(translation_ref)) + src_tocolmap.dot(translation_src) ) ])
    
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
        # as this is not related to camera poses, the rotations can be set to I. 
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


def organize_workspace(workspace: str, is_geometric: bool):
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
    organize_outputs(colmap_ws=depth_path, dst_folder=dst_folder, target_map = 'depth_maps', is_geom = is_geometric)
    organize_outputs(colmap_ws=depth_path, dst_folder=dst_folder, target_map = 'cost_maps', is_geom = is_geometric)
    
    
def organize_outputs(colmap_ws: str, dst_folder: str, target_map: str, is_geom: bool):
    '''
        It collect all target maps under the colmap_ws to dst_folder according to
        filenames.
        
        colmap_ws: str
            path to the colmap workspace
        
        dst_folder: str
            destination folder
            
        target_map: str
            maps to be reorganized, e.g. depth_map, cost_map and normal_map
    '''    
    if is_geom:
        map_type = "geometric"
    else:
        map_type = "photometric"
    
    all_maps_path = []
    all_views = sorted(glob.glob( os.path.join(colmap_ws, 'view*') ))
    for view in all_views:
        all_maps_path += glob.glob( os.path.join(view, 'stereo/' + target_map + '/*.{:s}.bin'.format(map_type)) )
    
    all_maps_path = sorted(all_maps_path)
    
    for image in all_maps_path:  
        if ('selected' in image): # all selected source views are not used for reconstruction
            continue
        
        image_name = image.split('/')[-1].split('.')[0][:-6]
        view_index = image.split('/')[-1].split('.')[0][-5:]
            
        dst_path = os.path.join(dst_folder, target_map + '/{:s}'.format(image_name)) 
        check_path_exist(dst_path)
        
        copyfile(image, os.path.join(dst_path, '{:s}_{:s}.{:s}.bin'.format( image_name, view_index, map_type))) 


def reconstruct_omni_maps(omni_workspace: str, Camera_parameter: list = None, 
                          view_to_syn: int = 6, resolution: list = None, 
                          maps_type: str = "depth_maps", save_omni: bool = True,
                          enable_geom: bool=False):
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
        
        enable_geom: bool
            Enable geometric filtering.
            
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
        path_to_file = glob.glob( os.path.join(path, view_name + '_view*') )
    
    cubemap = CubicMaps()
    cubemap.load_depthmap(path_to_file=path_to_file, type_ = map_type)
    
    if use_radial_dist:
        if camera_para is None:
            camera_para = [ cubemap.depthmap[0].shape[0]/2/np.tan(CUBICVIEW_FOV/2),    # fx
                            cubemap.depthmap[0].shape[0]/2/np.tan(CUBICVIEW_FOV/2),    # fy
                            cubemap.depthmap[0].shape[0]/2, cubemap.depthmap[0].shape[0]/2]    # cx, cy 
        elif len(camera_para) != 4:
            raise ValueError("Inpute ERROR. Camera parameters should have 4 parameters:[fx, fy, cx, cy].")
        
        for ind in range(len(path_to_file)):
            if map_type == 'depth_maps':
                cubemap.depthmap[ind] = cubemap.depth_trans(cubemap.depthmap[ind], camera_para)
            elif map_type == 'normal_maps':
                warnings.warn("Project normal maps are not supported by now.")
    
    if CUBICVIEW_FOV <= np.pi/2:
        cubemap.cube2sphere_fast(resolution = output_resolution, fov = CUBICVIEW_FOV)
    else:
        # think about cost maps
        cubemap.depthmap = viewsfusion(cubemap.depthmap, output_resolution)
    
    if save:
        if map_type == 'depth_maps' or 'cost_maps':
            cubemap.save_omnimage(path=path, name='360_'  + map_type + '.exr')
        else:
            cubemap.save_omnimage(path=path, name='360_'  + map_type + '.png')
    
    return cubemap.omnimage
    

def viewsfusion( depthList: list, resolution: tuple ):

    cam360List = []
    cubeTemp = CubicMaps()
    
    for ind, view in enumerate(depthList):
        depthTemp = [np.zeros(depthList[0].shape)] * 6
        depthTemp[ind] = depthList[ind]
        
        cubeTemp.depthmap = depthTemp
        cubeTemp.cube2sphere_fast(resolution = resolution, fov = CUBICVIEW_FOV)
        
        cam360Temp = Cam360(
                rotation_mtx = np.eye(3), 
                translation_vec = np.array([0,0,0]), 
                height = resolution[0], 
                width  = resolution[1], 
                channels = 1, 
                texture  = np.zeros(resolution), 
                depth    = cubeTemp.depth)        
        
        cam360List.append(cam360Temp)        
    
    method = 'sort'    
    output_depth = True
    rotation = np.eye(3)
    translation = np.array([8,8,1])
    Syn_view, Syn_depth = synthesize_view(cam360List, rotation, translation, 
                                          resolution, with_depth = output_depth, 
                                          method = method, parameters = 4)    
    return Syn_depth    


def evaluate(estimation: np.array, GT: np.array, checking_line: int = 100, max_: int = 25, error_threshold: int = 2, inf_value: int=50, save: bool = False,):   
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
            
        save: bool
            Whether to save the estimated depth map, ground truth as well as the difference map.
        
        max_: int
            The cutoff threshold for maximum values in figures and RMSE.
        
        Return:
        --------
            Raw RMSE and RMSE, the raw means that the depth errors from all 
            pixels are counted, including the sky and the ground.
        
    '''
    GT[GT>inf_value] = inf_value
    
    plt.figure(figsize=[12,12])
    min_d_toshow = min(estimation.min(), GT.min())
    max_d_toshow = max(estimation.max(), GT.max())

    # initialize error map and mask 
    diff_map = abs(estimation - GT)
    valid_mask = np.logical_and( GT > 0, GT < inf_value)
    
    # calculate rmse from valid pixels
    depth_score = np.sum(diff_map[valid_mask] > error_threshold)/valid_mask.sum() * 100
    print("There are {:.2f}% pixels having an error larger than {:.2f}.".format(depth_score, error_threshold))
    
    if not save:
        plt.subplot(421)
        plt.imshow(estimation, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow);
        plt.colorbar()
        plt.axis('off')
        plt.title('Estimated Depth')
        
        plt.subplot(422)
        plt.imshow(GT, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
        plt.colorbar()
        plt.axis('off')
        plt.title('Ground Truth')
        
        plt.subplot(423)
        plt.plot(GT[:,checking_line], '.');
        plt.plot(estimation[:,checking_line], '.');
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the vertical line at ' + str(checking_line) + 'th column')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
    
        plt.subplot(424)
        errors = abs(GT[:,checking_line] - estimation[:,checking_line])
        errors[errors > 20] = 0
        plt.plot(errors, '.')
        plt.xlabel('sky ------------------> ground \n Top to Bottom')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the vertical line at ' + str(checking_line) + 'th column')
        plt.grid()
        
        plt.subplot(425)
        plt.plot(GT[checking_line,:], '.');
        plt.plot(estimation[checking_line,:], '.');
        plt.xlabel('left ------------------> right \n ')
        plt.ylabel('Depth \n close ------------------> far')
        plt.title('Depth along the horizontal line at ' + str(checking_line) + 'th row')
        plt.legend(['ground truth', 'estimation'])
        plt.grid()
    
        plt.subplot(426)
        errors = abs(GT[checking_line,:] - estimation[checking_line,:])
        errors[errors > 20] = 0
        plt.plot(errors, '.')
        plt.xlabel('left ------------------> right')
        plt.ylabel('Absolute Error');
        plt.title('Absolute error along the horizontal line at ' + str(checking_line) + 'th row')
        plt.grid()
        plt.tight_layout()
        
        # plot error map
        plt.subplot(427)
        plt.imshow(np.abs(diff_map), cmap='RdYlGn_r', interpolation='nearest', vmin = 0, vmax = max_)
        plt.colorbar()
        plt.axis('off')
        plt.title('error map', fontsize=12);
        
        # plot error histogram
        plt.subplot(428)
        plt.hist(diff_map[valid_mask].flatten(), bins=50, histtype="stepfilled", density=True, alpha=0.6, range=(0, max_),cumulative=True)
        plt.grid(ls='--')
        plt.xlabel('errors', fontsize=12)
        plt.ylabel('cumulative percentage', fontsize=12)
        plt.title('histogram of errors', fontsize=12);
        
    else:   
        plt.figure()
        plt.imshow(estimation, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow);
        plt.colorbar()
        plt.axis('off')
        plt.savefig('estimated_depthmap.png', dpi=300, bbox_inches="tight")
        
        plt.figure()
        plt.imshow(GT, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
        plt.colorbar()
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
    
        # plot error map
        plt.figure(figsize=[8,4])
        plt.imshow(np.abs(diff_map), cmap='RdYlGn_r', interpolation='nearest', vmin = 0, vmax = max_)
        plt.colorbar()
        plt.axis('off')
        plt.savefig('Error_maps.png', dpi=300, bbox_inches="tight")
        
        # plot error histogram
        plt.figure(figsize=[8,4])
        plt.hist(diff_map[valid_mask].flatten(), bins=50, histtype="stepfilled", density=True, alpha=0.6, range=(0, max_), cumulative=True)
        plt.grid(ls='--')
        plt.xlabel('errors', fontsize=18)
        plt.ylabel('cumulative percentage', fontsize=18)
        plt.title('histogram of errors', fontsize=18);
        plt.savefig('hitrogram_error.png', dpi=300, bbox_inches="tight")
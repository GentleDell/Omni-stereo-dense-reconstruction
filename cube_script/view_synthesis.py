#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:03:45 2019

@author: zhantao

"""
import pickle

import cv2
import torch
import warnings
import numpy as np
from typing import Tuple
import spherelib as splib
import matplotlib.pyplot as plt

from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle

_SYN_METHODS_  = ['simple', 'sort', 'tv', 'median'] 

_WEIGHT_COSTS_  = 0.25
_WEIGHT_CENTER_ = 0.5
_WEIGHT_DISTS_  = 0.25

_NUM_CANDIDATES_  = 10

# cost function, affect the consistency of synthesized view
_COST_DEFINITION_ = 0   # 0: 'Gaussian';
                        # 1: 'Nearest' ;
# small number
EPS = 1e-8


def synthesize_view(cam360_list: list, rotation: np.array, translation: np.array, 
                    resolution: tuple, method: str = 'sort', parameters: float = 3.0, 
                    with_depth: bool = False, gpu_index: int = -1): 
    
    '''
        Given a list of cam360 objects, it synthesize a new view at the given pose with 
        the given resolution. 
        
        Firstly, it projects all pixels of given views to the synthesis view. Then it
        computes costs for candidates of each pixel and sort them with their costs.
        
        After the projection and aggregation, it filters outliers using the required
        methods. Finally, It gets txetures for the rest pixels.
        
        Parameters
        ----------  
        cam360_list: list
            source views for synthesis
            
        rotation: np.array
            rotation of the target view
        
        translation: np.array
            translation of the target view
        
        resolution: tuple
            resolution of the target view
        
        method: str
            filtering method to be used
            
        parameters: float
            parameters to be used in filters
            
        with_depth: bool
            whether to output depth map for the target view
        
        gpu_index: int
            the index of gpu to accelerate the function
            
    '''
    
    print("projecting pixels from original views to the synthesized view ...")
    projected_view = project_to_view(cam360_list, rotation, translation, resolution)
    
    print("computing the cost and aggregating pixels ...")
    pixels = aggregate_pixels(projected_view, resolution, fast = gpu_index>=0)
    
    print("conducting optimization ...")
    Syn_pixels = pixel_filter(pixels, resolution, method, parameter=parameters, gpu_ind=gpu_index)
    
    print("generating texture ...")
    texture = get_texture(Syn_pixels[:,[0,1,4,5]], cam360_list, resolution, with_depth)
    
    return texture


def pixel_filter(pixels: np.array, resolution: tuple, method: str, parameter: float, gpu_ind: int = -1):
    '''
        It filters out outliers using the required methods. Npw, 4 methods are 
        supported: 'simple', 'sort', 'tv', 'median'.
        
        'simple' keeps candidates from only one view and the view is set by 
        'parameter'.
        
        'sort' sorts all candidates according to their costs and keeps candidates
         having the smallest costs.

        'tv' and 'median' fiter on indices map. They treat the change of indices 
         as noises. So, for a patch on the synthesis view, these two filters tend 
         to use the texture from the same view. 
         
        Parameters
        ----------  
        pixels: np.array
            all candidates, each row contains:
                [nth_row_in_Syn, nth_col_in_Syn, depth, cost, 1D_coordinate_in_srcview, index_of_srcview]
                srcview means the view where this candidate comes from
        
        resolution: tuple
            resolution of the target view
            
        method: str
            filtering method to be used, including ['simple', 'sort', 'tv', 'median'] 
        
        parameters: float
            parameters to be used in filters.
                'simple' -- the src view to synthesize the view
                'sort'   -- None
                'tv'     -- lambda
                'median' -- size of the sliding window] 
        
        gpu_ind: int
            gpu index
            
    '''
    # verify the required method
    try:
        method_ind = _SYN_METHODS_.index(method.lower())
    except:
        warnings.warn('{:s} is not supported now; use "sort" to synthesize the view'.format(method))
        method_ind = 1
    
    if method_ind == 0:     # method:'simple'
        parameter = np.clip(np.round(parameter), pixels[:,-1].min(), pixels[:,-1].max())    # deal with invalid value
        Syn_pixels = pixels[pixels[:,5] == parameter,:]
    
    else:
        diff_ind = compute_diff(pixels[:,:2])
        top_pixels = pixels[diff_ind, :]
        
        if method_ind == 1:      # method:'sort'
            Syn_pixels = top_pixels
        else:
            # to use median filter or total variation filter, getting the indices map at first
            indice_image = -1 * np.ones(resolution)
            indice_image[top_pixels[:,0].astype(int), top_pixels[:,1].astype(int)] = top_pixels[:,5]
        
            if method_ind == 2:     # method:'tv'
                filtered_indices = denoise_tv_chambolle(indice_image, weight = parameter)
                
            elif method_ind == 3:   # method:'median'
                filtered_indices = median(indice_image.astype(np.uint8), disk(parameter))
                
            # synthesize the view
            func = [verify_and_synthesize_cpu, verify_and_synthesize_gpu]
            Syn_pixels = func[ gpu_ind >= 0 and torch.cuda.is_available() ](filtered_indices, indice_image, pixels, gpu_ind)
            
    return Syn_pixels


def verify_and_synthesize_cpu( new_indices: np.array, original_indices: np.array, all_pixels: np.array, gpu_ind = None ):
    '''
        It verifies whether the filtered indices are reasonable. Then it converts 
        verified indices to required formate.
        
        This is the cpu version and will be used if there is no GPU available or
        GPU is disabled. For a 512*1024 imag, it is 2 seconds slower then GPU
        version.
        
        Parameters
        ----------  
        new_indices: np.array
            the filtered indices
            
        original_indices: np.array
            indices before filtering
            
        all_pixels: np.array
            all candidates
            
        gpu_ind: None
            not used
            
    '''
    
    # Verify the filtered indices are reasonable.
    #    
    # It is required that a filtered view index for a certian pixel should have 
    # at least one candidate in the 'all_pixels' array. Otherwise, this index is 
    # invalid. 
    #
    # E.g. for the pixel at [255,255], all_pixel shows that view 7 and view 8 are 
    # two candidate views. If the filtered view index is one of the two view index, 
    # then it make sense to keep the index. However, if the filtered view is 3, 
    # not one of the two candidates, then the original view index will be resumed.
    #
    # Similarly, if there is no candidate, then the filtered index will be thrown.
    
    # convert the value (filtering will set -1 to be 255)
    new_indices = new_indices.astype(int)
    new_indices[new_indices == 255] = -1
    index_of_view = np.unique(all_pixels[:,-1])
    
    # count the number of candidates for every pixel and save it into a 3D matrix
    # together with information of the coordinates and view indices.
    keys, counts = np.unique(all_pixels[:, [0,1,5]], axis=0, return_counts=True)
    compress_pix_cnt = np.zeros( new_indices.shape + (index_of_view.max().astype(int)+2,) )  
         # '+2' creates another layer to deal with -1 for resume_flag (-1 is treated as 'the last element')
         # '+1' works as well but it will slightly affect the verification process.
    compress_pix_cnt[keys[:,0].astype(int), keys[:,1].astype(int), keys[:,2].astype(int)] = counts
    
    # get the pixels whose view indices were changed during filtering; get the 
    # corresponding indices.
    diff_row, diff_col = np.where(new_indices - original_indices != 0)
    diff_view_index = new_indices[diff_row, diff_col]
    
    # if there is at least one candidates for a pixel, keep it. (not necessary 
    # to include the filtered indices)
    keep_flag = np.sum(compress_pix_cnt[diff_row, diff_col, :], axis = 1) > 0
    # if the filtered indices do not have corresponding candidates, resume the 
    # original indices
    resume_flag = compress_pix_cnt[diff_row, diff_col, diff_view_index] == 0
    
    new_indices[diff_row, diff_col] = (1-resume_flag)*new_indices[diff_row, diff_col] + resume_flag*original_indices[diff_row, diff_col]
    new_indices[diff_row, diff_col] = keep_flag*new_indices[diff_row, diff_col] + -1*(1 - keep_flag)
    
    # Convert verified indices to required formate
    # 
    # Firstly, get pixel coordinate, raveled position and view index for all candidates.
    # As it is possible that for some pixels, there are many candidates from the same view
    # and the number of the candidates is not fixed. Meanwhile, it is complicated to create
    # a variant-length array. So, we only keep the best candidates from each view for each
    # pixel. 
    #
    # This is consistent with the original_indices, because it is generated by keeping
    # the best cabdidate for each pixel.
    # 
    # Then, the view is synthesized by taking pixel coordinates, raveled positions and
    # view indices for all pixels of the synthesis view.
    
    # flip to inverse order to keep the best candidate for each pixel and view (as the 
    # all_pixels is sorted descendingly with costs)
    keys = np.flip(all_pixels[:, [0,1,4,5]], axis=0)
    compress_ravel_pos = np.zeros( new_indices.shape + (len(index_of_view),) )
    compress_ravel_pos[keys[:,0].astype(int), keys[:,1].astype(int), keys[:,3].astype(int)] = keys[:,2]
    
    # taking pixel coordinates, raveled positions and view indices for synthesis view
    pix_coord = np.where(new_indices >= 0)
    view_ind = new_indices[pix_coord]
    
    # formating the data
    Syn_pixels = np.zeros([len(pix_coord[0]), 6])
    ravel_pos  = compress_ravel_pos[pix_coord[0], pix_coord[1], view_ind]
    Syn_pixels[:,:2] = np.vstack(pix_coord).T
    Syn_pixels[:,4] = ravel_pos
    Syn_pixels[:,5] = view_ind
    
    return Syn_pixels


def verify_and_synthesize_gpu( new_indices: np.array, original_indices: np.array, all_pixels: np.array, gpu_ind: int = 0):
    '''
        It verifies whether the filtered indices are reasonable. Then it converts 
        verified indices to required formate.
        
        GPU accelerated verision; as pytorch on cpu is much slower than numpy, 
        we use the above verify_and_synthesize_cpu for cpu and this one for gpu.
        As a result, there are some redundent codes. 
        For 512*1024 images, 2 seconds faster.
        
        Parameters
        ----------  
        new_indices: np.array
            the filtered indices
            
        original_indices: np.array
            indices before filtering
            
        all_pixels: np.array
            all candidates
            
        gpu_ind: int
            gpu index
            
    '''
    
    # Verify the filtered indices are reasonable.
    #    
    # It is required that a filtered view index for a certian pixel should have 
    # at least one candidate in the 'all_pixels' array. Otherwise, this index is 
    # invalid. 
    #
    # E.g. for the pixel at [255,255], all_pixel shows that view 7 and view 8 are 
    # two candidate views. If the filtered view index is one of the two view index, 
    # then it make sense to keep the index. However, if the filtered view is 3, 
    # not one of the two candidates, then the original view index will be resumed.
    #
    # Similarly, if there is no candidate, then the filtered index will be thrown.
    
    all_pixels_gpu = torch.from_numpy(all_pixels).cuda(gpu_ind)
    
    # convert the value (filtering will set -1 to be 255)
    new_indices = new_indices.astype(int)
    new_indices[new_indices == 255] = -1
    index_of_view = torch.unique(all_pixels_gpu[:,-1])
    
    # count the number of candidates for every pixel and save it into a 3D matrix
    # together with information of the coordinates and view indices.
    keys, counts = torch.unique(all_pixels_gpu[:, [0,1,5]], dim=0, return_counts=True)
    compress_pix_cnt = torch.zeros( new_indices.shape + (int(index_of_view.max())+2,) ).long().cuda(gpu_ind) 
         # '+2' creates another layer to deal with -1 for resume_flag (-1 is treated as 'the last element')
         # '+1' works as well but it will slightly affect the verification process.
    keys = keys.long()
    compress_pix_cnt[keys[:,0], keys[:,1], keys[:,2]] = counts
    
    # get the pixels whose view indices were changed during filtering; get the 
    # corresponding indices.
    diff_row, diff_col = np.where(new_indices - original_indices != 0)
    diff_view_index = new_indices[diff_row, diff_col]
    
    # if there is at least one candidates for a pixel, keep it. (not necessary 
    # to include the filtered indices)
    keep_flag = (torch.sum(compress_pix_cnt[diff_row, diff_col, :], dim = 1) > 0).cpu().numpy()
    # if the filtered indices do not have corresponding candidates, resume the 
    # original indices
    resume_flag = (compress_pix_cnt[diff_row, diff_col, diff_view_index] == 0).cpu().numpy()
    
    new_indices[diff_row, diff_col] = (1-resume_flag)*new_indices[diff_row, diff_col] + resume_flag*original_indices[diff_row, diff_col]
    new_indices[diff_row, diff_col] = keep_flag*new_indices[diff_row, diff_col] + -1*(1 - keep_flag)
    
    # Convert verified indices to required formate
    # 
    # Firstly, get pixel coordinate, raveled position and view index for all candidates.
    # As it is possible that for some pixels, there are many candidates from the same view
    # and the number of the candidates is not fixed. Meanwhile, it is complicated to create
    # a variant-length array. So, we only keep the best candidates from each view for each
    # pixel. 
    #
    # This is consistent with the original_indices, because it is generated by keeping
    # the best cabdidate for each pixel.
    # 
    # Then, the view is synthesized by taking pixel coordinates, raveled positions and
    # view indices for all pixels of the synthesis view.
    
    # flip to inverse order to keep the best candidate for each pixel and view (as the 
    # all_pixels is sorted descendingly with costs)
    keys = torch.from_numpy( np.flip(all_pixels[:, [0,1,4,5]], axis=0).astype(int) ).cuda(gpu_ind)
    compress_ravel_pos = torch.zeros( new_indices.shape + (len(index_of_view),) ).long().cuda(gpu_ind)
    compress_ravel_pos[keys[:,0], keys[:,1], keys[:,3]] = keys[:,2]
    
    # taking pixel coordinates, raveled positions and view indices for synthesis view
    pix_coord = np.where(new_indices >= 0)
    view_ind = new_indices[pix_coord]
    
    # formating the data
    Syn_pixels = np.zeros([len(pix_coord[0]), 6])
    ravel_pos  = compress_ravel_pos[pix_coord[0], pix_coord[1], view_ind]
    Syn_pixels[:,:2] = np.vstack(pix_coord).T
    Syn_pixels[:,4] = ravel_pos.cpu().numpy()
    Syn_pixels[:,5] = view_ind
    
    return Syn_pixels


def get_texture(texture_index: np.array, cam360_list: list, resolution: tuple, with_depth: bool):
    '''
        It get textures and depthmap for the synthesis view from the given source
        views.
        
        Parameters
        ----------  
        texture_index: np.array
            the indices of textures of the synthesis view
            format: [row of synthesis view,      (0 - resolution [0])
                     column of synthesis view,   (0 - resolution [1])
                     index of the source pixel,  (0 - resolution [0]*resolution [1])
                     index of the view from which the source pixel is picked ]  [0 - len(cam360_list)]]
        
        cam360_list: list
            source views for synthesis
        
        resolution: tuple
            resolution of the target view
            
        with_depth: bool
            whether to output depth map for the target view
            
    '''
    # initialize the synthesis view
    syn_view = np.zeros(resolution + (3,))
    
    # for each view, to get texture/depth, the data are vecotorized at first. 
    # Then the data corresponding to the texture_index are taken.
    for ind in range(len(cam360_list)):
        
        cam = cam360_list[ind]
        
        pixel_index = texture_index[ texture_index[:,-1] == ind ].astype(int)
        cam_ravel = cam.texture.reshape([-1,3])
        
        syn_view[ pixel_index[:, 0], pixel_index[:, 1],: ] = cam_ravel[ pixel_index[:,2], : ]
        
    if with_depth:
        # initialize the synthesis depth map and cost map
        depth_view = np.zeros(resolution)
        cost_view  = np.zeros(resolution)
        
        for ind in range(len(cam360_list)):
        
            cam = cam360_list[ind]
            
            pixel_index = texture_index[ texture_index[:,-1] == ind  ].astype(int)
            dep_ravel = cam.depth.reshape([-1])
            cost_ravel= cam.cost.reshape([-1])
            
            depth_view[ pixel_index[:, 0], pixel_index[:, 1] ] = dep_ravel[ pixel_index[:,2] ]
            cost_view [ pixel_index[:, 0], pixel_index[:, 1] ] = cost_ravel[ pixel_index[:,2] ]
            
        return syn_view, depth_view, cost_view
        
    return syn_view


def project_to_view(cam360_list: list, rotation: np.array, translation: np.array, resolution: tuple):
    '''
        It projects the given cam360 list to the view with the given pose and resolution.
        
        Parameters
        ----------  
        cam360_list: list
            source views for synthesis
            
        rotation: np.array
            rotation of the target view
        
        translation: np.array
            translation of the target view
        
        resolution: tuple
            resolution of the target view
            
    '''
    # initialize an array to store projections
    projections = np.empty((0, 6))
    for ind in range(len(cam360_list)): 
        
        print(">>>> projecting the {:d} view ...".format(ind))
        
        cam = cam360_list[ind]
        
        # Build an equiangular grid in (theta, phi) coordinates.
        theta, phi = cam.get_sampling_axes()
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # Vectorize the computed equiangular grid.
        theta = theta_grid.ravel()
        phi = phi_grid.ravel()
        
        # distortion can not be visualized by reshape(as we should take theta 
        # and phi into consideration)
        theta_new, phi_new, depth_new = project(
                theta, phi, cam.depth.ravel(),
                cam._rotation_mtx, cam._translation_vec, cam._radius,
                rotation, translation, rad_b=1)
        
        # convert theta and phi into pixel coordinate
        theta_pix = theta_new *  resolution[0]/np.pi
        phi_pix   = phi_new * resolution[1]/(2*np.pi)
        
        # compute a basic cost in this step to reduce computation in further steps.
        # Here the basic costs include costs from colmap and the costs of distances
        # between the projected points and the center of pixels [proportional to the
        # distance between 3D points and rays from camera center to pixels centers]
        basic_cost= _WEIGHT_COSTS_  * cam.cost.reshape(-1) + \
                    _WEIGHT_CENTER_ * np.sqrt( (theta_pix - np.round(theta_pix))**2 + (phi_pix - np.round(phi_pix))**2 ) 
        
        # the indices of pixels in raveled image
        pixel_ind = np.arange(0, cam._height * cam._width)
        
        # the index of the view from which these pixels come
        view_ind  = ind*np.ones((cam._height * cam._width))
    
        # all views are concatenated into a matrix of (res[0], num_view*res[1], 6) 
        # where [:,:,:2] is the index of row and column of corresponding pixel in 
        # synthesis view; [:,:,2] is depth; [:,:,3] is cost; [:,:,4] is the index
        # in the raveled source view, [:,:,5] is the index of view from which these
        # pixels come.
        projected_depth = np.stack((np.clip( np.floor(theta_pix), 0, resolution[0]-1) ,  
                                    np.clip( np.floor(phi_pix)  , 0, resolution[1]-1) , 
                                    depth_new, 
                                    basic_cost,
                                    pixel_ind,
                                    view_ind), axis = 1)
        
        # remove invalid depth, which is -1; 
        # then use percentile to remove outliers;
        projected_depth = projected_depth[ projected_depth[:, 2] != -1, : ]
        minPercentile = np.percentile(projected_depth, 5, axis = 0)[2]
        projected_depth = projected_depth[ projected_depth[:, 2] > minPercentile + EPS, : ]
        
        # expand the depth value of pxiel to a 3 by 3 cross, so that more holes
        # and invalid values can be removed.
        #                _
        #    _         _|_|_ 
        #   |_|  ==>  |_|_|_|  
        #               |_|
        #
        top, bottom, left, right = ( np.zeros(projected_depth.shape),np.zeros(projected_depth.shape), 
                                     np.zeros(projected_depth.shape), np.zeros(projected_depth.shape) )
        
        # do not touch the boundary
        top    [ projected_depth[:,0] > 0, 0] = -1
        bottom [ projected_depth[:,0] < resolution[0] - 1, 0 ] = 1
        
        left   [ projected_depth[:,1] > 0, 1] = -1
        right  [ projected_depth[:,1] < resolution[1] - 1, 1] = 1
        
        projected_depth = np.vstack( (projected_depth, 
                                      projected_depth+top,
                                      projected_depth+left,
                                      projected_depth+right,
                                      projected_depth+bottom))
        
        # append the expanded and projected depth maps to projections matrix
        projections = np.append(projections, projected_depth, axis = 0)        
    
    return projections 


def aggregate_pixels(projections: np.array, resolution: tuple, fast: bool = False):
    '''
        For each pixel of the synthesized view, it computes costs for all candidates.
        
        Parameters
        ----------    
        projections: np.array
            An array containing all candidates for sythesis.
            
        resolution: tuple
            Resolution of the synthesized view.
            
    '''
    # sort by the index of row and column, depth and then cost
    sort_ind = np.lexsort( (projections[:,3], projections[:,2], projections[:,1], projections[:,0]) )
    projections = projections[sort_ind]
    
    if fast:
        projections = compute_cost_fast(projections, resolution)
    else:
        projections = compute_cost(projections, resolution)
    
    # sort by the index of row, column as well as costs
    sort_ind = np.lexsort( (projections[:,3], projections[:,1], projections[:,0]) )
    projections = projections[sort_ind]

    return projections


def compute_diff(coordinates: np.array):
    '''
        It returns the index of difference on coordinates
    '''
    row_ind = np.diff(coordinates[:,0]) > 0 
    col_ind = np.diff(coordinates[:,1]) > 0
    diff_ind = np.where( row_ind | col_ind )[0] + 1
    diff_ind = np.insert(diff_ind, 0, 0)
    
    return diff_ind


def compute_cost(depth_cost_array: np.array, resolution: tuple):
    '''
        It computes the final costs with the given depths and basic costs.
        
        For a pixel, if there is only one candidate then its costs won't be
        changed. But if there are more than one candidates, the final costs 
        for these candidates will be a weighted sum of standardized depths 
        of these candidates and the corresponding basic costs. The Formulation 
        is:
            cost = _W1_ * basic_costs + _W2_ * std_depth
            
        For the standardization, Guassian assumption and nearest neigbor are 
        supported now. For the Gaussian assumption, the closer to the the mean
        the lower of the costs. But for the nearest neigbor, the closer to the 
        camera center the lower of the costs. Which meothd to use is decided by 
        pre-defined global variable ---- _COST_DEFINITION_ 
            0: 'Gaussian',
            1: 'Nearest' ;
        
        This implemetation is a slow version, but it is guaranteed to find the
        best candidates for pixels (as long as they has candidates).
        
        Parameters
        ----------  
        depth_cost_array: np.array
            array containing coordinates [:, :2], depth[:, 3] and cost[:, 4]
            
        resolution: tuple
            the resolution of the final image.
            
    '''    
    # get the indices where coordinates change
    diff_ind = compute_diff(depth_cost_array[:,:2])
    valid_range = len(diff_ind)-1

    # calculate the cost by group
    for ind in range(valid_range):
        
        depth = depth_cost_array[ diff_ind[ind]:diff_ind[ind+1], 2]     # get a group of depth
        norm_depth = (depth - depth.mean())/(depth.var() + 1e-10)       # standardize the group of depth
        
        # costs are weighted sum of basic costs and standardized depth
        basic_cost = depth_cost_array[diff_ind[ind]:diff_ind[ind+1], 3]
        cost  = (_WEIGHT_DISTS_ * norm_depth + basic_cost) * _COST_DEFINITION_ \
               +(_WEIGHT_DISTS_ * np.abs(norm_depth) + basic_cost ) * (1 - _COST_DEFINITION_)
        
        depth_cost_array[diff_ind[ind]:diff_ind[ind+1], 3] = cost
        
    
    return depth_cost_array


def compute_cost_fast(depth_cost_array: np.array, resolution: tuple):
    '''
        It computes the final costs with the given depths and basic costs.
        
        The idea is the same as the function above, but to accelerate the 
        execution, only the first _NUM_CANDIDATES_ candidates of each pixel 
        are considered to compute the final costs.
     
        As a result, the best candidate is not guaranted to be found by this
        function. but it still perfomance well excepth for some inconsistency
        on the final textures and depth. 
        
        For a 512*1024 image, it costs about 5 second to compute the final 
        costs -- 8 seconds faster then the slower one.
        
        depth_cost_array: np.array
            all candidates
            
        resolution: tuple
            output resolution
            
    '''    
    pixel_array = np.nan * np.ones(resolution + (_NUM_CANDIDATES_,))
    view_array  = np.nan * np.ones(resolution + (_NUM_CANDIDATES_,))
    cost_array  = np.nan * np.ones(resolution + (_NUM_CANDIDATES_,))
    depth_array = np.nan * np.ones(resolution + (_NUM_CANDIDATES_,))
    
    # save the top 10 candidates for each pixel
    ind = 0
    while ind < _NUM_CANDIDATES_ and depth_cost_array.shape[0] > 0:
        
        # get candidates for pixels
        diff_ind = compute_diff(depth_cost_array[:,:2])
        
        # save corresponding data
        coordinate = depth_cost_array[diff_ind, :2].astype(int)
        depth_array[coordinate[:,0], coordinate[:,1], ind] = depth_cost_array[diff_ind, 2]
        cost_array[coordinate[:, 0], coordinate[:,1], ind] = depth_cost_array[diff_ind, 3]
        pixel_array[coordinate[:,0], coordinate[:,1], ind] = depth_cost_array[diff_ind, 4]
        view_array[coordinate[:, 0], coordinate[:,1], ind] = depth_cost_array[diff_ind, 5]
        
        # remove recorded data
        depth_cost_array = np.delete(depth_cost_array, diff_ind, axis = 0)
        
        ind += 1
    
    # standardize depth 
    depth_array[np.isnan( depth_array[:,:,0]),0] = -1
    mean = np.repeat(np.nanmean(depth_array, axis=2)[:,:,np.newaxis], _NUM_CANDIDATES_, axis=2)
    var  = np.repeat( (np.nanvar(depth_array, axis=2) + 1e-10)[:,:,np.newaxis], _NUM_CANDIDATES_, axis=2)
    std_depth = (depth_array - mean)/var
    
    # costs are weighted sum of basic costs and standardized depth
    cost_array = (_WEIGHT_DISTS_ * std_depth + cost_array) * _COST_DEFINITION_ \
                +(_WEIGHT_DISTS_ * np.abs(std_depth) + cost_array ) * (1 - _COST_DEFINITION_)
    
    # organize the data format
    depth_array[np.isnan(depth_array)] = -1
    row, col, layer= np.where(depth_array > 0)
    
    depth_cost_array = np.zeros([len(row), 6])
    depth_cost_array = np.vstack((row, 
                                 col, 
                                 depth_array[row, col, layer], 
                                 cost_array[ row, col, layer], 
                                 pixel_array[row, col, layer],
                                 view_array[ row, col, layer])).T
    return depth_cost_array
    

def project(theta: np.array, phi: np.array, depth: np.array,
            rot_mtx_a: np.array, t_vec_a: np.array, rad_a: float,
            rot_mtx_b: np.array, t_vec_b: np.array, rad_b: float) -> Tuple[np.array, np.array, np.array]:
    """
    It projects the coordinate (theta, phi) in the sphere A to the sphere B. It computes the derivative of each
    component, if required.
    The rotation matrices and the translation vectors refer to a common world coordinate system.
    (Adapted from spherelib.projection)

    Args:
        theta: elevation coordinate (N,) in the sphere A.
        phi: azimuth coordinate (N,) in the sphere A.
        depth: depth values (N,) associated to the coordinate (theta, phi) in the sphere A.
        rot_mtx_a: rotation matrix (3, 3) from the world coordinate system to the one of the sphere A.
        t_vec_a: translation vector (3,) from the coordinate system of the sphere A to the world one.
        rad_a: radius of the sphere A.
        rot_mtx_b: rotation matrix (3, 3) from the world coordinate system to the one of the sphere B.
        t_vec_b: translation vector (3,) from the coordinate system of the sphere B to the world one.
        rad_b: radius of the sphere B.

    Returns:
        theta_proj: the corresponding elevation (N,) on the sphere B.
        phi_proj: the corresponding azimuth (N,) on the sphere B.
        
    """

    # Compute the 3D coordinate of theta and phi on the sphere A.
    x, y, z = splib.pol2eu(theta, phi, rad_a)
    f = np.stack((x, y, z), axis=0)

    # Compute auxiliary variables.
    a = rot_mtx_b.dot(rot_mtx_a.T) * (rad_b / rad_a)
    b = (- rot_mtx_b.dot(rot_mtx_a.T).dot(t_vec_a) + t_vec_b) * rad_b

    # Compute the projection on the sphere B in 3D coordinates.
    a_f = a.dot(f)                               # spherical point clond under the rotation of b
    num = (a_f * depth) + b[:, None]

    # Compute the projection on the sphere B in polar coordinate.
    theta_proj, phi_proj, depth_proj = splib.eu2pol(num[0, :], num[1, :], num[2, :])
    # theta_proj, phi_proj are (N,) np.arrays.

    return theta_proj, phi_proj, depth_proj


def evaluation( texture: np.array, texture_GT: np.array, depth:np.array, depth_GT: np.array):
    min_d_toshow = min(depth.min(), depth_GT.min())
    max_d_toshow = max(depth.max(), depth_GT.max())

    plt.imshow(texture);
    plt.axis('off')
    plt.savefig('synthesized_texture.png', dpi=300, bbox_inches="tight")
    
    plt.imshow(texture_GT)
    plt.axis('off')
    plt.savefig('Groundtruth_texture.png', dpi=300, bbox_inches="tight")

    plt.imshow(depth, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow);
    plt.axis('off')
    plt.savefig('estimated_depthmap.png', dpi=300, bbox_inches="tight")
    
    plt.imshow(depth_GT, cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
    plt.axis('off')
    plt.savefig('Groundtruth_depthmap.png', dpi=300, bbox_inches="tight")
    
    plt.imshow(abs(depth_GT - depth), cmap = 'magma', vmin=min_d_toshow, vmax=max_d_toshow)
    plt.colorbar()
    plt.axis('off')
    plt.savefig('error_maps.png', dpi=300, bbox_inches="tight")

    diff_map = abs(depth_GT - depth)
    mask = depth!=depth.min()
    masked_RMSE = np.sqrt(np.sum(diff_map[mask]**2)/(np.sum(mask)))
    print('The masked RMSE for depth is:', masked_RMSE )
    
    diff_texture = np.abs( cv2.cvtColor(texture_GT.astype('uint8'), cv2.COLOR_RGB2GRAY) - cv2.cvtColor((texture*255).astype('uint8'), cv2.COLOR_RGB2GRAY))
    rmse = np.sqrt(np.sum(diff_texture[mask]**2)/(np.sum(mask)))
    psnr = 20 * np.log10(255 / rmse)
    print('The masked PSNR for texture is:', psnr )
    
    print("{:f}% pixels are filled".format(np.sum(mask)/diff_map.size*100))


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar  
    from: 
        https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113 @ Greenstick
        
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
    
#    printProgressBar(0, valid_range, prefix = 'Progress:', suffix = 'Complete', length = 50)
#    printProgressBar(ind + 1, valid_range, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        
        
        
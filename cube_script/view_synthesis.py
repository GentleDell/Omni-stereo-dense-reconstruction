#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:03:45 2019

@author: zhantao

"""
import pickle

import warnings
import numpy as np
from typing import Tuple
import spherelib as splib
import matplotlib.pyplot as plt

from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle

_SYN_METHODS_  = ['easy', 'sort', 'tv', 'median'] 

_WEIGHT_COSTS_  = 0.25
_WEIGHT_CENTER_ = 0.5
_WEIGHT_DISTS_  = 0.25

_COST_DEFINITION_ = 0   # 0: 'Gaussian';
                        # 1: 'Nearest' ;


def synthesize_view(cam360_list: list, rotation: np.array, translation: np.array, 
                    resolution: tuple, method: str = 'sort', parameters: int = 3, with_depth = False): 
    
    '''
        Given a list of cam360_list, it synthesize a new view at the given pose with 
        the given resolution. 
    '''
    
#    print("projecting pixels of original views to the synthesized view ...")
#    projected_view = project_to_view(cam360_list, rotation, translation, resolution)
#    
#    print("computing the cost and aggregating pixels ...")
#    pixels = aggregate_pixels(projected_view, resolution)
    
    pickle_in = open("temp.pickle","rb")
    pixels = pickle.load(pickle_in)
    
    print("conducting optimization ...")
    Syn_pixels = pixel_filter(pixels, resolution, method, parameter=parameters)
    
    print("generating texture ...")
    texture = get_texture(Syn_pixels[:,[0,1,4,5]], cam360_list, resolution)
    
    return texture


def pixel_filter(pixels: np.array, resolution: tuple, method: str, parameter: float):
        
    try:
        method_ind = _SYN_METHODS_.index(method.lower())
    except:
        warnings.warn('{:s} is not supported now; use "sort" to synthesize the view'.format(method))
        method_ind = 1
    
    if method_ind == 0:
        parameter = np.clip(np.round(parameter), pixels[:,-1].min(), pixels[:,-1].max())
        Syn_pixels = pixels[pixels[:,5] == parameter,:]
    
    else:
        diff_ind = compute_diff(pixels[:,:2])
        top_pixels = pixels[diff_ind, :]
        
        if method_ind == 1:
            Syn_pixels = top_pixels
        else:
            
            indice_image = -1 * np.ones(resolution)
            indice_image[top_pixels[:,0].astype(int), top_pixels[:,1].astype(int)] = top_pixels[:,5]
        
            if method_ind == 2:
                filtered_indices = denoise_tv_chambolle(indice_image, weight = parameter)
                
            elif method_ind == 3:
                filtered_indices = median(indice_image.astype(np.uint8), disk(parameter))
    
            Syn_pixels = verify_and_synthesize( filtered_indices, indice_image, pixels )
            
    return Syn_pixels


def verify_and_synthesize( new_indices: np.array, original_indices: np.array, all_pixels: np.array ):
    
    
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
#    TODO: solve -1
    index_of_view = np.unique(all_pixels[:,-1])
    
    # count the number of candidates for every pixel and save it into a 3D matrix
    # together with information of the coordinates and view indices.
    keys, counts = np.unique(all_pixels[:, [0,1,5]], axis=0, return_counts=True)
         # '+2' creates another layer to deal with -1 in line 124 (treated as 'the last element')
         # '+1' works as well but it will slightly affact the verification process.
    compress_pix_cnt = np.zeros( new_indices.shape + (index_of_view.max().astype(int)+2,) )  
    compress_pix_cnt[keys[:,0].astype(int), keys[:,1].astype(int), keys[:,2].astype(int)] = counts
    
    # get the pixels whose view indices were changed during filtering; get the 
    # corresponding new indices.
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
    # and the number of the candidates is not fixed. Meawhile, it is complicate to create
    # an variant-length array. So, we only keep the best candidates from each view for each
    # pixel. 
    #
    # This is consistent with the original_indices, because it is generated by keeping
    # the best cadidates for each pixel.
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


def get_texture(texture_index: np.array, cam360_list: list, resolution: tuple):
    
    syn_view = np.zeros(resolution + (3,))
    
    for ind in range(len(cam360_list)):
        
        cam = cam360_list[ind]
        
        pixel_index = texture_index[ texture_index[:,-1] == ind  ].astype(int)
        cam_ravel = cam.texture.reshape([-1,3])
        
        syn_view[ pixel_index[:, 0], pixel_index[:, 1],: ] = cam_ravel[ pixel_index[:,2], : ]
        
    return syn_view


def project_to_view(cam360_list: list, rotation: np.array, translation: np.array, resolution: tuple):
    
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
        
        basic_cost= _WEIGHT_COSTS_  * cam.cost.reshape(-1) + \
                    _WEIGHT_CENTER_ * np.sqrt( (theta_pix - np.round(theta_pix))**2 + (phi_pix - np.round(phi_pix))**2 ) 
        
        pixel_ind = np.arange(0, cam._height * cam._width)
        
        view_ind  = ind*np.ones((cam._height * cam._width))
        
        projections = np.append(projections, np.stack((np.clip( np.round(theta_pix), 0, 511) ,  
                                                       np.clip( np.round(phi_pix),  0, 1023) , 
                                                       depth_new, 
                                                       basic_cost,
                                                       pixel_ind,
                                                       view_ind), axis = 1), axis = 0)
        # all views are concatenated into a matrix of (res[0], num_view*res[1], 4) 
        # where [:,:,:2] is the row and column of corresponding pixel in synthesis 
        # view while [:,:,2] is the depth and [:,:,3] is the cost
    
    return projections 


def aggregate_pixels(projections: np.array, resolution: tuple):
    '''
        For each pixel of the synthesized view, it computes costs for all candidates.
        
        Parameters
        ----------    
        projections: np.array
            An array containing all candidates for sythesis.
            
        resolution: tuple
            Resolution of the synthesized view.
    '''
    
    sort_ind = np.lexsort( (projections[:,1], projections[:,0]) )
    projections = projections[sort_ind]
    
    projections[:,3] = compute_cost(projections[:,:4], resolution)
    
    sort_ind = np.lexsort( (projections[:,3], projections[:,1], projections[:,0]) )
    projections = projections[sort_ind]

    return projections


def compute_diff(coordinates: np.array):
    
    row_ind = np.diff(coordinates[:,0]) > 0 
    col_ind = np.diff(coordinates[:,1]) > 0
    diff_ind = np.where( row_ind | col_ind )[0] + 1
    diff_ind = np.insert(diff_ind, 0, 0)
    
    return diff_ind


def compute_cost(depth_cost_array: np.array, resolution: tuple):
    '''
        It computes the cost with the given depth and cost tensor.
        
        If there are many estimations, the cost will be computed with normal distribution 
        or the nearest depth, with a predifined weight.
    '''
    keys, counts = np.unique(depth_cost_array[:, [0,1,5]], axis=0, return_counts=True)
    compress_pix_depth = np.zeros( resolution + (depth_cost_array[:,-1].max().astype(int)+1,) )  
    compress_pix_depth[keys[:,0].astype(int), keys[:,1].astype(int), keys[:,2].astype(int)] = counts
    
    
#    diff_ind = compute_diff(depth_cost_array[:,:2])
#    valid_range = len(diff_ind)-1
#    
#
#    for ind in range(valid_range):
#        
#        depth = depth_cost_array[ diff_ind[ind]:diff_ind[ind+1], 2]
#        norm_depth = (depth - depth.mean())/(depth.var() + 1e-10)
#        
#        basic_cost = depth_cost_array[diff_ind[ind]:diff_ind[ind+1], 3]
#        cost  = (_WEIGHT_DISTS_ * norm_depth + basic_cost) * _COST_DEFINITION_ \
#               +(_WEIGHT_DISTS_ * np.abs(norm_depth) + basic_cost ) * (1 - _COST_DEFINITION_)
#        
#        depth_cost_array[diff_ind[ind]:diff_ind[ind+1], 3] = cost
        
    return depth_cost_array[:,3]
    

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
    num = (a_f * depth) - b[:, None]             # 3D point cloud under coordinate b (was a bug which was '+ b[:, None]')

    # Compute the projection on the sphere B in polar coordinate.
    theta_proj, phi_proj, depth_proj = splib.eu2pol(num[0, :], num[1, :], num[2, :])
    # theta_proj, phi_proj are (N,) np.arrays.

    return theta_proj, phi_proj, depth_proj


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
        
        
        
        
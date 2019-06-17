#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:03:45 2019

@author: zhantao

"""
import torch
import pickle
import itertools
import numpy as np
from typing import Tuple
import spherelib as splib
import matplotlib.pyplot as plt

_TOP_N_PIXELS_ = 10

_WEIGHT_COSTS_  = 0.25
_WEIGHT_CENTER_ = 0.5
_WEIGHT_DISTS_  = 0.25

_COST_DEFINITION_ = 0   # 0: 'Gaussian';
                        # 1: 'Nearest' ;


def synthesize_view(cam360_list: list, rotation: np.array, translation: np.array, 
                    resolution: tuple, with_depth = False, gpu_index: int = 0): 
        
#    projected_view = project_to_view(cam360_list, rotation, translation, 
#                                resolution)
    
#    coord_vol = aggregate_pixels(projected_view, resolution, gpu_index)
    
    # saved in this TEST
    pickle_in = open("./synthesis_cam360.pickle","rb")
    coord_vol = pickle.load(pickle_in)
    
    pixels = np.array(list(itertools.zip_longest(*coord_vol, fillvalue=0))).T
    best_pixels = np.stack(pixels[:,0])
    
    texture = get_texture(best_pixels)
    
    return texture


def get_texture(texture_index: np.array):
    
    
    
    return texture


def project_to_view(cam360_list: list, rotation: np.array, translation: np.array, resolution: tuple):
    
    projections = np.empty((resolution[0], 0, 4))
    for cam in cam360_list: 
        # here requires all 360 cams to be the same size
        
        # 360 image size
        height = cam._height
        width  = cam._width
        
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
        theta_new = theta_new.reshape((height, width)) *  resolution[0]/np.pi
        phi_new   = phi_new.reshape((height, width)) * resolution[1]/(2*np.pi)
        
        depth_new = depth_new.reshape((height, width))
        basic_cost= _WEIGHT_COSTS_  * cam.cost + \
                    _WEIGHT_CENTER_ * np.sqrt( (theta_new - np.round(theta_new))**2 + (phi_new - np.round(phi_new))**2 ) 
    
        projections = np.append(projections, np.stack((np.round(theta_new), np.round(phi_new), depth_new, basic_cost), axis = 2), axis = 1)
        # all views are concatenated into a matrix of (res[0], num_view*res[1], 4) 
        # where [:,:,:2] is the row and column of corresponding pixel in synthesis 
        # view while [:,:,2] is the depth and [:,:,3] is the cost
    
    return projections 


def aggregate_pixels(projections: np.array, resolution: tuple, gpu_index: int):
    '''
        For each pixel of the synthesized view, it computes costs for all candidates
        and keeps top _TOP_N_PIXELS_ pixels for further operation.
        
        Parameters
        ----------    
        projections: np.array
            An array containing all candidates for sythesis.
            
        resolution: tuple
            Resolution of the synthesized view.
            
        gpu_index: int
            The index of gpu for synthesis.
    '''
    coord_vol = []
    
    # obtain data for synthesis view
    projections_torch = torch.from_numpy(projections)
    if torch.cuda.is_available():
        projections_torch = projections_torch.cuda(gpu_index)
    
    for row in range( 0, 10 ):
        print('sweeping the {:d} row'.format(row))
        for col in range(resolution[1]):
            mask = round_to_pixel(projections_torch, (row,col))
            cost = compute_cost(projections_torch[mask][:,2:])                  
            
            valid_size = min(_TOP_N_PIXELS_, len(cost))
            sorted_cost, sort_ind = torch.topk( cost, k = valid_size, largest = False )
            
            coord_vol.append( mask.nonzero()[sort_ind].cpu().numpy() )

    return coord_vol
    

def compute_cost(depth_cost_array: torch.tensor):
    '''
        It computes the cost with the given depth and cost tensor.
    '''
    # if there are many estimations, the cost will be computed with normal distribution 
    # or the nearest depth, with a predifined weight. 
    depth = depth_cost_array[:,0]
    norm_depth = (depth - torch.mean(depth)) / (torch.std(depth, unbiased = False) + 1e-10)
    
    # if _COST_DEFINITION_ is 1, the smaller of the distance the lower of the cost from depth
    # if _COST_DEFINITION_ is 1, the cost will be computed as gaussian distribution
    cost  = (_WEIGHT_DISTS_ * norm_depth + depth_cost_array[:,1]) * _COST_DEFINITION_ \
           +(_WEIGHT_DISTS_ * torch.abs(norm_depth) + depth_cost_array[:,1] )* (1 - _COST_DEFINITION_)
           
    return cost


def round_to_pixel(projected_mat, target: tuple):    
    
    mask_theta = projected_mat[:,:,0] == target[0]
    mask_phi   = projected_mat[:,:,1] == target[1]
    
    mask = mask_theta * mask_phi
    
    return mask
    

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
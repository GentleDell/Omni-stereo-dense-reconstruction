#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:35:03 2020

@author: zhantao
"""

import numpy as np
from numpy import array

from bilateralSolver import bilateral_solver
from normalizedCrossCorrelation import NCC

# maximum cost from colmap
MAXCOST = 2 - 1e-6


def normalFilter(depth : array, normal : array, cost : array, 
                 angular_threshold: float = 45/180*np.pi, depth_threshold: float = 50):
    """
    Filtering pixels of the depth map and cost map, whose normal is different from nearby pixels.

    Parameters
    ----------
    depth :  array
        Input depth array.
    normal :  array
        Input noraml array.
    cost :  array
        Input cost array.
    angular_threshold : float, optional
        The maximum angluar differnece between pixels. The default is 45/180*np.pi.
    depth_threshold : float, optional
        The maximum depth of each pixel.

    Returns
    -------
    depthOut : array
        Filtered depth map.
    costOut : array
        filtered cost map.

    """
    
    rowNormalConsis = np.zeros(normal.shape[:2])
    for row in range(1,normal.shape[0]-1):
        rowNormalConsis[row-1,:] = np.logical_and(np.sum(normal[row,:,:] * normal[row-1,:,:], axis = 1) > np.cos(angular_threshold)
                                                , np.sum(normal[row,:,:] * normal[row+1,:,:], axis = 1) > np.cos(angular_threshold) ) 
    colNormalConsis = np.zeros(normal.shape[:2])
    for col in range(1,normal.shape[0]-1):
        colNormalConsis[:,col-1] = np.logical_and(np.sum(normal[:,col,:] * normal[:,col-1,:], axis = 1) > np.cos(angular_threshold), 
                                                  np.sum(normal[:,col,:] * normal[:,col+1,:], axis = 1) > np.cos(angular_threshold) ) 
    maskValidNormal = np.linalg.norm(normal, axis = 2) > 0
    
    mask = maskValidNormal*rowNormalConsis*colNormalConsis
    
    depthOut = depth.copy()
    depthOut[ mask < 1 ] = 0
    depthOut[depthOut > depth_threshold] = 0 
    depthOut[depthOut < 0] = 0 
    
    costOut  = np.clip(cost, 0, MAXCOST)
    costOut[mask < 1] = MAXCOST
    costOut[depthOut > depth_threshold] = MAXCOST
    costOut[depthOut < 0] = MAXCOST
    
    return depthOut, costOut


def postProcess(refImage : array, depth : array, normal : array, cost : array, refPose : array, refcam : array,
                srcImage : array, srcPose : array, srccam : array, patchSize : tuple = (5, 5), 
                normalDiffThreshold : float = 60/180*np.pi, depthThreshold : float = 50, verifyProcess : bool = False):
    """
    It removes depth outliers according to the normal consistency and then it
    uses bilateral solver to recover depth. Finally, it computes the NCC for 
    each pixel/patch as confidence for view fusion.

    Parameters
    ----------
    refImage : array
        Reference view to compute the NCC.
    depth : array
        Depth map of the reference view.
    normal : array
        Input noraml map.
    cost : array
        Input cost map.
    refPose : array
        Pose of the reference view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    refcam : array
        Camera parameters of the reference view, i.e. [fx, fy, cx, cy].
    srcImage : array
        Source view to compute the NCC.
    srcPose : array
        Pose of the source view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    srccam : array
        Camera parameters of the source view, i.e. [fx, fy, cx, cy].
    patchSize : tuple, optional
        Size of patches. The default is (5,5).
    normalDiffThreshold : float, optional
        The maximum angluar differnece between pixels. The default is 60/180*np.pi.
    depthThreshold : float, optional
        The maximum depth of each pixel. The default is 50.
    verifyProcess : bool, optional
        Whether to verify some intermediate results. The default is False.

   Returns
    -------
    depthFill : array
        Filtered and filled depth map.
    NccMap : array
        The computed NCC for each pixel.
        Its size is [row - patchSize[0] + 1, col - patchSize + 1].

    """

    # filter out outliers having inconsistent normal 
    depth, cost = normalFilter(depth, normal, cost, normalDiffThreshold, depthThreshold)

    # use bilateral filter to inpaint depth map
    depthFill = bilateral_solver(depth, cost, refImage)

    # compute the normalized cross correlation as confidence map for fusion
    NccMap = NCC(refImage, refPose, refcam, srcImage, srcPose, srccam, depthFill, patchSize, verifyProcess)
    
    return depthFill, NccMap
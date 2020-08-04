#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  19 18:39:57 2019

@author: zhantao
"""
import os
import glob
import subprocess
from shutil import rmtree
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam360 import Cam360
from cubicmaps import CubicMaps
from bilateralSolver import bilateral_solver
from view_synthesis import aggregate_pixels, pixel_filter, synthesize_view



def normalFilter(depth: np.array, normal: np.array, cost: np.array, 
                 angular_threshold: float = 45/180*np.pi, depth_threshold: float = 7):
    
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


def viewsfusion_noProp( depthList: list, costList: list, resolution: tuple ):

    cubeTemp = CubicMaps()
    delta_row = 2*np.tan(CUBICVIEW_FOV/2)/resolution[0]
    delta_col = 2*np.tan(CUBICVIEW_FOV/2)/resolution[0]
    
    projections = np.empty((0, 6))
    for ind, view in enumerate(depthList):
        points, mask_face  = cubeTemp.spherical2img(ind, resolution, CUBICVIEW_FOV)
        row = np.round(points[:,0]/delta_row - 0.5).astype(int)
        col = np.round(points[:,1]/delta_col - 0.5).astype(int)
        mask_face = mask_face.reshape(resolution)
        
        depthVec  = view[row, col].ravel()
        costVec   = costList[ind][row, col].ravel()
        pixels    = np.nonzero(mask_face)
        thetaVec  = pixels[0]
        phiVec    = pixels[1]
        pixel_ind = np.array(row * resolution[0] + col)
        view_ind  = ind*np.ones(pixel_ind.shape)
        
        projected_depth = np.stack( (thetaVec, phiVec, depthVec, costVec, pixel_ind, view_ind), axis = 1)
        
        # remove invalid depth, which is -1;
        projected_depth = projected_depth[ projected_depth[:, 2] > 0, : ]
        
        # append the expanded and projected depth maps to projections matrix
        projections = np.append(projections, projected_depth, axis = 0)        

    print("computing the cost and aggregating pixels ...")
    pixels = aggregate_pixels(projections, resolution, fast = False)
    
    print("conducting filtering ...")
    method = 'sort'
    Syn_pixels = pixel_filter(pixels, resolution, method=method, parameter=1)
    
    print("select depth...")
    Syn_depth = np.zeros(resolution)
    Syn_cost  = np.zeros(resolution)

    Syn_depth[ Syn_pixels[:, 0].astype(int), Syn_pixels[:, 1].astype(int) ] = Syn_pixels[:, 2]
    Syn_cost [ Syn_pixels[:, 0].astype(int), Syn_pixels[:, 1].astype(int) ] = Syn_pixels[:, 3]
    
    return np.expand_dims(Syn_depth, axis=2), np.expand_dims(Syn_cost, axis=2)

    
def viewsfusion_Prop( depthList: list, costList: list, resolution: tuple ):

    cubeTemp = CubicMaps()
    delta_row = 2*np.tan(CUBICVIEW_FOV/2)/resolution[0]
    delta_col = 2*np.tan(CUBICVIEW_FOV/2)/resolution[0]
    
    projections = np.empty((0, 6))
    for ind, view in enumerate(depthList):
        points, mask_face  = cubeTemp.spherical2img(ind, resolution, CUBICVIEW_FOV)
        row = np.round(points[:,0]/delta_row - 0.5).astype(int)
        col = np.round(points[:,1]/delta_col - 0.5).astype(int)
        mask_face = mask_face.reshape(resolution)
        
        depthVec  = view[row, col].ravel()
        costVec   = costList[ind][row, col].ravel()
        pixels    = np.nonzero(mask_face)
        thetaVec  = pixels[0]
        phiVec    = pixels[1]
        pixel_ind = np.array(row * resolution[0] + col)
        view_ind  = ind*np.ones(pixel_ind.shape)
        
        projected_depth = np.stack( (thetaVec, phiVec, depthVec, costVec, pixel_ind, view_ind), axis = 1)
                
        # expand the depth value of a pxiel to the closest 2 by 2 grid, so that holes
        # and invalid values could be removed.
        #              _ _
        #    _        |_|_|
        #   |_|  ==>  |_|_|
        #    
        top, left, right, bottom, topleft, topright, bottomleft, bottomright = (
                np.zeros(projected_depth.shape),np.zeros(projected_depth.shape), 
                np.zeros(projected_depth.shape),np.zeros(projected_depth.shape),
                np.zeros(projected_depth.shape),np.zeros(projected_depth.shape), 
                np.zeros(projected_depth.shape),np.zeros(projected_depth.shape))
        
        # pixels to be expanded
        maskTop   = points[:,0]/delta_row - 0.5 - row < 0
        maskLeft  = points[:,1]/delta_col - 0.5 - col < 0
        maskRight = points[:,1]/delta_col - 0.5 - col > 0
        maskBottom= points[:,0]/delta_row - 0.5 - row > 0
        maskTopLeft  = np.logical_and(maskTop, maskLeft)
        maskTopRight = np.logical_and(maskTop, maskRight)
        maskBottomLeft  = np.logical_and(maskBottom, maskLeft)
        maskBottomRight = np.logical_and(maskBottom, maskRight)
       
        # image boundaries
        topboundary  = projected_depth[:, 0] <= 0
        leftboundary = projected_depth[:, 1] <= 0
        rightboundary  = projected_depth[:, 1] >= resolution[1]-1
        bottomboundary = projected_depth[:, 0] >= resolution[0]-1
    
        # compute the costs of these 'expanded' depth value
        maskTop = np.logical_and(maskTop, ~topboundary)
        top[ maskTop, 0 ] = -1
        
        maskBottom = np.logical_and(maskBottom, ~bottomboundary)
        bottom[ maskBottom, 0 ] = 1
        
        maskLeft = np.logical_and(maskLeft, ~leftboundary) 
        left[ maskLeft, 1] = -1
        
        maskRight = np.logical_and(maskRight, ~rightboundary)
        right[maskRight, 1] = 1
        
        maskTopLeft = np.logical_and(maskTopLeft,  ~(topboundary+leftboundary))
        topleft[maskTopLeft, 0] = -1
        topleft[maskTopLeft, 1] = -1
        
        maskTopRight = np.logical_and(maskTopRight, ~(topboundary+rightboundary))
        topright[maskTopRight, 0] = -1
        topright[maskTopRight, 1] = +1
        
        maskBottomLeft = np.logical_and(maskBottomLeft,  ~(bottomboundary+leftboundary))
        bottomleft[maskBottomLeft, 0] = +1 
        bottomleft[maskBottomLeft, 1] = -1
        
        maskBottomRight = np.logical_and(maskBottomRight, ~(bottomboundary+rightboundary))
        bottomright[maskBottomRight, 0] = +1
        bottomright[maskBottomRight, 1] = +1
        
        projected_depth = np.vstack( (projected_depth, 
                                      (projected_depth+top)[maskTop, :],
                                      (projected_depth+left)[maskLeft, :],
                                      (projected_depth+right)[maskRight, :],
                                      (projected_depth+bottom)[maskBottom, :],
                                      (projected_depth+topleft)[maskTopLeft, :],
                                      (projected_depth+topright)[maskTopRight, :],
                                      (projected_depth+bottomleft)[maskBottomLeft, :],
                                      (projected_depth+bottomright)[maskBottomRight, :]))
        
        # remove invalid depth, which is -1; then use percentile to remove outliers;
        projected_depth = projected_depth[ projected_depth[:, 2] > 0, : ]
        
        # append the expanded and projected depth maps to projections matrix
        projections = np.append(projections, projected_depth, axis = 0)        

    print("computing the cost and aggregating pixels ...")
    pixels = aggregate_pixels(projections, resolution, fast = False)
    
    print("conducting filtering ...")
    method = 'sort'
    Syn_pixels = pixel_filter(pixels, resolution, method=method, parameter=1)
    
    print("select depth...")
    Syn_depth = np.zeros(resolution)
    Syn_cost  = np.zeros(resolution)

    Syn_depth[ Syn_pixels[:, 0].astype(int), Syn_pixels[:, 1].astype(int) ] = Syn_pixels[:, 2]
    Syn_cost [ Syn_pixels[:, 0].astype(int), Syn_pixels[:, 1].astype(int) ] = Syn_pixels[:, 3]
    
    return np.expand_dims(Syn_depth, axis=2), np.expand_dims(Syn_cost, axis=2)


def viewsfusion_old( depthList: list, costList: list, resolution: tuple ):

    cam360List = []
    cubeTemp = CubicMaps()
    
    rotation = np.eye(3)
    translation = np.array([0, 0, 0])
    for ind, view in enumerate(depthList):
        omniDepth = np.zeros(resolution)   
        omniCost  = np.ones(resolution) * 5    # default cost should be a large value 
        
        depthTemp = [np.zeros(depthList[0].shape)] * 6
        depthTemp[ind] = depthList[ind]
        
        costTemp  = [np.ones(costList[0].shape) * 5] * 6
        costTemp[ind] = costList[ind]
        
        omniDepth = cubeTemp.cube2sphere_fast( depthTemp, resolution, fov = CUBICVIEW_FOV )[:,:,0]
        omniCost  = cubeTemp.cube2sphere_fast( costTemp, resolution, fov = CUBICVIEW_FOV )[:,:,0]
        
        cam360Temp = Cam360(
                rotation_mtx = rotation, 
                translation_vec = translation, 
                height = resolution[0], 
                width  = resolution[1], 
                channels = 3, 
                texture  = np.zeros(resolution + (3,)), 
                depth    = omniDepth,
                cost     = omniCost)        
        
        cam360List.append(cam360Temp)        
    
    method = 'sort'    
    output_depth = True
    Syn_view, Syn_depth, Syn_cost = synthesize_view(cam360List, rotation, translation, 
                                          resolution, with_depth = output_depth, 
                                          method = method, parameters = 4)    
    return np.expand_dims(Syn_depth, axis=2), np.expand_dims(Syn_cost, axis=2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:34:21 2020

@author: zhantao
"""

import numpy as np
import open3d as o3d
from numpy import array
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.transform import Rotation
from interpolation.splines import CubicSplines
from matplotlib import pyplot as plt


_EPS_NCC_      = 1e-8
_SPLINE_ORDER_ = 3

def visualizePoints(points : array, color : array = None):
    """
    Visualize the given points cloud with the given color.

    Parameters
    ----------
    points : array
        Input point cloud, shape = (N, 2).
    color : array
        Input color cloud, shape = (N, 3). The default is None.

    Returns
    -------
    None.

    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[::8,:])
    
    if color is not None:
        if color.max() > 1:
            normValue = 255
        else:
            normValue = 1
        pcd.colors = o3d.utility.Vector3dVector(color[::8,:]/normValue)
        
    o3d.visualization.draw_geometries([pcd, mesh_frame])


def patchify(image : array, patch_shape : array) -> tuple:
    """
    Patchify the given image according to the given patch_size. 

    Parameters
    ----------
    image : array
        Input image array with a shape of (row, col, channel).
    patch_shape : array
        The shape of each patch. 

    Returns
    -------
    imagePatches : array
        An ndarray of the image patches, with a shape of:
            (row-patch_shape + 1, col-patch_shape + 1, patch_shape, patch_shape, channel).
            
    """
    image = np.ascontiguousarray(image)
    X,Y,Z = image.shape
    x, y  = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z)
    strides = image.itemsize* array([Z*Y, Z, Z*Y, Z, 1])
    imagePatches = as_strided(image, shape=shape, strides=strides)
    
    return imagePatches


def projectPatches(depthPatches : array, uvPatches : array,
                   refPose : array, refcam : array, 
                   srcPose : array, srccam : array, 
                   visPoints : bool = False) -> array:
    """
    It projects reference patches to the source image according to the corresponding 
    depth patches, uv-coordinate patches, camera intrinsics and extrinsics.

    Parameters
    ----------
    depthPatches : array
        Input depth patches of the reference image.
    uvPatches : array
        uv delta coordinprojectPatchesate patches corresponding to the depth patches.
    refPose : array
        Pose of the reference view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    refcam : array
        Camera parameters of the reference view, i.e. [fx, fy, cx, cy].s
    srcPose : array
        Pose of the source view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    srccam : array
        Camera parameters of the source view, i.e. [fx, fy, cx, cy].
    visPoints : bool
        Whether visulize the projected point cloud. The default is False.

    Returns
    -------
    array
        Pixels of the source view corresponding to the reference patches.
        shape = (2, depthPatches.size)
        
    """
    
    Z = depthPatches
    
    X , Y  = uvPatches[:,:,:,:,0][:,:,:,:,None] * Z/refcam[0], uvPatches[:,:,:,:,1][:,:,:,:,None] * Z/refcam[1]   
    points = array([X.flatten(), Y.flatten(), Z.flatten()])

    quaternion= [tmp for tmp in refPose[1:4]] + [refPose[0]]
    Rot_w2ref = Rotation.from_quat(quaternion).as_matrix()
    tra_w2ref = refPose[4:]
    
    quaternion= [tmp for tmp in srcPose[1:4]] + [srcPose[0]]
    Rot_w2src = Rotation.from_quat(quaternion).as_matrix()
    tra_w2src = srcPose[4:]
    
    worldPoint= Rot_w2ref.transpose()@points - Rot_w2ref.transpose() @ tra_w2ref[:,None]
    srcPoints = Rot_w2src @ worldPoint + tra_w2src[:,None]  
    
    srcProjMat= array([[srccam[0], 0, srccam[2]],
                       [0, srccam[1], srccam[3]],
                       [0,         0,        1]])
    srcPixels = srcProjMat @ srcPoints
    srcPixels = (srcPixels/(srcPixels[2,:] + _EPS_NCC_))[:2, :]
   
    if visPoints:
        # only when there are not too much points
        if np.prod(list(points.shape)) * 3 * 4 / np.power(2, 20) < 500:
            
            step = 3
            
            centerHalfStride = ((np.prod(depthPatches.shape[2:4])+1)/2).astype(int)
            
            refPoints = points.transpose()[ centerHalfStride :: 2*centerHalfStride - 1, :]
            wrdPoints = worldPoint.transpose()[ centerHalfStride :: 2*centerHalfStride - 1, :]
            socPoints = srcPoints.transpose()[ centerHalfStride :: 2*centerHalfStride - 1, :]
            pointCloud= np.vstack((refPoints[::step, :], wrdPoints[::step, :], socPoints[::step, :]))
            
            print('Only show the close (<7) points.')
            print('Black is refpoints; drak gray is world points; gray is points under src corrdinate.')
            pointCloud[pointCloud > 7] = 0
            pointCloud[pointCloud < -7] = 0
            
            size = refPoints[::step, :].shape[0]
            colorCloud= np.vstack(((0.2*np.ones([size,3])),
                                   (0.5*np.ones([size,3])),
                                   (0.7*np.ones([size,3]))))
            visualizePoints(pointCloud, colorCloud)
    
    return srcPixels


def cubicSpline(image : array) -> CubicSplines:
    """
    It generates a cubic spline for the given image.

    Parameters
    ----------
    image : array
        The image to have the cubic spline.

    Returns
    -------
    CubicSplines
        The generated cubic spline.

    """
    
    padding = _SPLINE_ORDER_ + 1
    
    # Pad the top and bottom parts of the image by repeating the first and last rows, respectively.
    texture_wrapped = np.concatenate(
        (np.tile(image[0, :, :], (padding, 1, 1)),
         image,
         np.tile(image[-1, :, :], (padding, 1, 1))), axis=0)
    
    # Pad the left and right sides of the image.
    texture_wrapped = np.concatenate(
        (np.tile(texture_wrapped[:, 0, :][:,None,:], (1, padding, 1)),
         texture_wrapped,
         np.tile(texture_wrapped[:,-1, :][:,None,:], (1, padding, 1))), axis=1)
    
    # Compute the first and last value of column.
    # The axis is referred to as 'extended' as it takes the vertical padding into account.
    extended_col_first_value = 0.5 - padding
    extended_col_last_value  = image.shape[1] - 0.5 + padding

    # Compute the first and last value of row.
    # The axis is referred to as 'extended' as it takes the horizontal padding into account.
    extended_row_first_value = 0.5 - padding
    extended_row_last_value  = image.shape[0] - 0.5 + padding

    # Compute the lengths of the extended theta and phi axes.
    extended_row_length = texture_wrapped.shape[0]
    extended_col_length = texture_wrapped.shape[1]
    
    # Create a spline approximation of the texture.
    low = [extended_col_first_value, extended_row_first_value]
    up = [extended_col_last_value, extended_row_last_value]
    orders = [extended_row_length, extended_col_length]
    image_spline = CubicSplines(
        low, up, orders,
        np.reshape(texture_wrapped, (texture_wrapped.shape[0] * texture_wrapped.shape[1], image.shape[2])))
    
    return image_spline


def normalizePatches(patches : array) -> array :
    """
    It normalizes the given patches array. Then it returns the vectorized and 
    normalized patches.

    Parameters
    ----------
    patches : array
        Patches to be normalized.

    Returns
    -------
    array 
        The vectorized and normalized patches..

    """
    patchVec  = patches.reshape(patches.shape[0], patches.shape[1], -1)
    
    patchMean = np.mean(patchVec, axis = 2)
    patchStd  = np.std(patchVec , axis = 2)
    
    normPatch = (patchVec - patchMean[:,:,None])/(patchStd[:,:,None] + _EPS_NCC_)
    
    return normPatch


def diffPatches(refPatches : array, srcPatches : array) -> array :
    
    refVec = refPatches.reshape(refPatches.shape[0], refPatches.shape[1], -1)
    srcVec = srcPatches.reshape(srcPatches.shape[0], srcPatches.shape[1], -1)

    difVec = np.abs(refVec - srcVec)
    
    oneVec = np.ones([1, 1, refVec.shape[2]])
    disVec = np.sum(difVec * oneVec, axis = 2)
    nomDis = disVec/np.linalg.norm(difVec, axis = 2)/np.linalg.norm(oneVec, axis = 2)
    
    return nomDis


def NCC(refView : array, refPose : array, refcam : array,
        srcView : array, srcPose : array, srccam : array, 
        depth : array, patchSize : tuple = (5, 5), 
        verifyPatches : bool = False) -> array:
    """
    Given views, cameras and corresponding poses, compute the normalized cross 
    correlation between pixels as the confidence.

    Parameters
    ----------
    refView : array
        Reference view to compute the NCC.
    refPose : array
        Pose of the reference view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    refcam : array
        Camera parameters of the reference view, i.e. [fx, fy, cx, cy].
    srcView : array
        Source view to compute the NCC.
    srcPose : array
        Pose of the source view, i.e. [qw, qx, qy, qz, tx, ty, tz].
    srccam : array
        Camera parameters of the source view, i.e. [fx, fy, cx, cy].
    depth : array
        Depth map of the reference view.
    patchSize : tuple, optional
        Size of patches. The default is (5,5).
    verifyPatches : bool
        Whether to verify the patches. The default is True.
        
    Returns
    -------
    NccMap : array
        The computed NCC for each pixel.
        Its size is [row - patchSize[0] + 1, col - patchSize + 1]

    """
    refPatches = patchify(refView, patchSize)
    depPatches = patchify(depth[:,:,None], patchSize)
    depPatches = depPatches[:,:,2,2,0][:,:,None,None,None] * np.ones(depPatches.shape)
    uvIndices  = array( np.meshgrid( np.arange(0.5, refView.shape[1]),
                                     np.arange(0.5, refView.shape[0]) )).swapaxes(0, 1).swapaxes(1,2) - refcam[2:]
    uvPatches  = patchify(uvIndices, patchSize)
    
    srcPixels  = projectPatches(depPatches, uvPatches, refPose, refcam, srcPose, srccam, verifyPatches)
    outPixels  = (srcPixels[0, :] >= srcView.shape[1]) + (srcPixels[1, :] >= srcView.shape[0]) + (srcPixels[0, :] <  0) + (srcPixels[1, :] < 0)
    
    srcCubSpln = cubicSpline(srcView)
    srcTexture = srcCubSpln.interpolate( np.flip(srcPixels, 0).transpose(), diff=False).clip(0,255).astype(int)
    srcTexture[outPixels, :] = 0
    
    srcPatches = srcTexture.flatten().reshape(list(uvPatches.shape[:-1]) + [3]).swapaxes(2,3)
    
    rowPatches, colPatches = refPatches.shape[:2]
    
    if verifyPatches:
        
        print('To verify the patches, it would be better to use the ground truth depth map.')
        
        centerHalfStride = ((np.prod(patchSize)+1)/2).astype(int)
        centerSrcPatches = srcTexture[ centerHalfStride :: 2*centerHalfStride - 1, :]
        reconsSrcImage   = centerSrcPatches.reshape(rowPatches, colPatches, 3)
        
        refTexture = refPatches.reshape(-1, 3)
        centerRefPatches = refTexture[ centerHalfStride :: 2*centerHalfStride - 1, :]
        reconsRefImage   = centerRefPatches.reshape(rowPatches, colPatches, 3)
        
        plt.figure(figsize = [12,8])
        plt.subplot(2,2,1)
        plt.imshow(srcView)
        plt.title('src image')
        plt.subplot(2,2,2)
        plt.imshow(reconsSrcImage)
        plt.title('reconstructed from \nsrc patches center')
        plt.subplot(2,2,3)
        plt.imshow(refView)
        plt.title('ref image')
        plt.subplot(2,2,4)
        plt.imshow(reconsRefImage)
        plt.title('reconstructed from \nref patches center')
        plt.tight_layout()
        plt.show()
    
    # Normalied cross correlation
    refNormVecPatch = normalizePatches(refPatches)
    srcNormVecPatch = normalizePatches(srcPatches)
    NccMap = np.sum(refNormVecPatch*srcNormVecPatch, axis = 2)/np.prod(patchSize)/refView.shape[2]
        
    # Vector difference
    # NccMap = diffPatches(refPatches, srcPatches)
    
    
    padding = 2
    # Pad the top and bottom parts of the image by repeating the first and last rows, respectively.
    NccMap_padded = np.concatenate(
        (np.tile(NccMap[0, :], (padding, 1)),
         NccMap,
         np.tile(NccMap[-1, :], (padding, 1))), axis=0)
    
    # Pad the left and right sides of the image.
    NccMap_padded = np.concatenate(
        (np.tile(NccMap_padded[:, 0][:,None], (1, padding)),
         NccMap_padded,
         np.tile(NccMap_padded[:,-1][:,None], (1, padding))), axis=1)
    
    return NccMap_padded
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:36:17 2019

@author: zhantao
"""
import cv2
import warnings
import numpy as np

from cam360 import Cam360
from cubicmaps import CubicMaps

def view_selection(cam:Cam360, reference:np.array, initial_pose: tuple, fov: tuple=(np.pi/2, np.pi/2),
                   max_iter: int=10, use_filter: bool=True, threshold: float=5.0):
    '''
        It selects the view that maximize the overlapping between the referece 
        image and the source view projected from the given cam360 obj. \n
        
        Similar to the colmap, sparse features are used to select views. At each loop: \n
        1. Features in the reference image and the source view are detected and matched. 
        2. For features in each view (two views in total at each loop), the centroid
           is calculated. Then the distance between the two centroids are calculated. 
        3. Accoding to the distance between the two centroids, a better view is projected 
           as a new source view.
        4. Repeating 1-3 until the distance bwteen the two centroids are smaller than 
           the given threshold.
        
        
        Parameters
        ----------    
        cam : Cam360
            The source cam360 obj from which the source view is projected.
            
        reference : np.array
            The reference image.
            
        initial_pose : tuple
            The first pose to start the search.
            
        fov: tuple
            Field of view of the selected view. 
            
        use_filter: bool
            Whether to use filters to filter out wrong (unreliable) matches;
            Here we use fundamental matrix to do the filtering. 
            
        max_iter: int
            Maximum iterations.
            
        threshold: float (in pixel)
            The threshold to decide whether the reference view and the source 
            view are close enough.
        
        Examples
        --------
        >>> img, pose = view_selection(cam360, ref_view, 
                                       initial_pose=(np.pi*3/2, np.pi/2))
    '''
    # initialize objs to be used
    orb = cv2.ORB_create()
    cubemaps = CubicMaps()
    
    # the size of a pixel
    unit_x = 1/reference.shape[1]
    unit_y = 1/reference.shape[0]
    
    # color image to grayscale image
    if reference.max() <= 1:
        reference = np.round(reference*255)
    reference = cv2.cvtColor(reference.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    pose = initial_pose
    for cnt in range(max_iter):
        
        # obtain a source view
        source = cubemaps.cube_projection(cam=cam, direction=(pose + fov), resolution=reference.shape)
        source = np.round(source*255)    
        source = cv2.cvtColor(source.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
        # feature detection and matching
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(source, None)
    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # filtering outliers 
        if use_filter:
            matches = filter_matches(kp1, kp2, matches)
        if len(matches) <= 5:
            warnings.warn("Can not find enough inlier matches")
            return None, None
        
        # calculate the distance between centroids
        centroid_dist = feature_centroid_distance(kp1, kp2, matches, source.shape)
                
        if np.sqrt(np.sum(centroid_dist**2)) <= threshold:
            break
        else:
            # update poses
            delta_phi = -np.arctan(centroid_dist[0]*unit_x)
            delta_theta = -np.arctan(centroid_dist[1]*unit_y)
            pose = (pose[0] + delta_phi, pose[1] + delta_theta)
        
    return source, pose    


def filter_matches(kp1, kp2, matches):
    '''
        It uses RANSAC and fundamental matrix provided by opencv to filter out 
        outliers.
    '''
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3.0)
    matches_filttered = [ matches[m] for m in np.where(mask[:,0]==1)[0].tolist() ]
    
    return matches_filttered


def feature_centroid_distance(kp1, kp2, matches, image_size):
    '''
        For each group of keypoints, it calculates a weighted centroid and then it computes
        the distance between the two weighted centroids.
    '''
    # reference image center
    center = (image_size[0]/2, image_size[1]/2)
    
    # positions of keypoints
    ref_features = np.array([kp1[m.queryIdx].pt for m in matches])  # features on the reference image
    src_features = np.array([kp2[m.trainIdx].pt for m in matches])  # features on the source image
    
    # the weight is based on the distance between ref_features and the image center
    weight = np.sqrt(np.sum((ref_features-center)**2, axis=1))
    weight = weight/sum(weight)
    
    # computer the weighted centroids
    ref_centroid = np.average(ref_features, weights = weight, axis = 0)
    src_centroid = np.average(src_features, weights = weight, axis = 0)
    
    return ref_centroid - src_centroid
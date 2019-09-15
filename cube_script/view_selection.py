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
import matplotlib.pyplot as plt

# thresholds to sort candidate views
MIN_NUM_FEATURE = 20
MIN_OVERLAPPING = 40
MIN_TRIANGULATION = 6*np.pi/180

UPDATE_RATE = 1.0
    

def view_selection(cam:Cam360, reference:np.array, initial_pose: tuple, fov: tuple=(np.pi/2, np.pi/2),
                   max_iter: int=10, use_filter: bool=True, threshold: float=10.0):
    '''
        It selects a view together with its score that satisfies two criterions:
            1. sharing the maximum overlapping between the referece image and 
               the source view projected from the given cam360 obj. (measured
               by the number of matches and must be greater than a threshold) \n
            2. the trangulation angle must be greater than a threshold, otherwise 
               the socre of this view will have a low score.
               
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
            The first pose to start the search. (theta, phi)
            
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
        
        Return
        --------
        source: np.array
            selected view
            
        pose: tuple:
            (theta, phi)
            
        score: float
            score of the selected view
        
        Examples
        --------
        >>> img, pose = view_selection(cam360, ref_view, 
                                       initial_pose=(np.pi/2, np.pi*3/2))
    '''
    # initialize objs to be used
    orb = cv2.ORB_create()
    cubemaps = CubicMaps()
    
    # color image to grayscale image
    if reference.max() <= 1:
        reference = np.round(reference*255)
    reference = cv2.cvtColor(reference.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    # ATTENTION, here the pose is filipped. [becomes (phi, theta)]
    initial_pose = (initial_pose[1], initial_pose[0])
    
#    unit_x = 1/reference.shape[1]
#    unit_y = 1/reference.shape[0]
    
    pose = initial_pose
    phi, theta = pose[0], pose[1]
    for cnt in range(max_iter):
        
        # obtain a source view
        source = cubemaps.cube_projection(cam=cam, direction=(pose + fov), resolution=reference.shape)
        source_gray = cv2.cvtColor( np.round(source*255).astype('uint8'), cv2.COLOR_RGB2GRAY )
       
        ##################################
#        DEMO AND DEBUG
#        plt.imshow(source)
#        plt.axis('off')
#        plt.savefig("view_{:d}.png".format(cnt), bbox_inches='tight')
#     cubemaps.cube_projection(cam=cam, direction=((3.8,1.57) + fov), resolution=reference.shape)
        ##################################
        
        # feature detection and matching
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(source_gray, None)
        
#       draw_keypoints(reference, kp1)
#       draw_keypoints(source_gray, kp2)
    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # filtering outliers 
        if use_filter:
            matches = filter_matches(kp1, kp2, matches)
        if len(matches) <= MIN_NUM_FEATURE:                 # Can not find enough inlier matches; start random searching
            theta, phi = None, None
            pose = (np.random.normal(loc=initial_pose[0], scale=1.0),
                    np.random.normal(loc=initial_pose[1], scale=0.5))
            pose = correct_angles(pose)
            continue
        
        ####################            
#        DEMO AND DEBUG
#        show_matches = cv2.drawMatches(reference, kp1, source_gray, kp2, matches[:20], None, flags=2)
#        plt.imshow(show_matches)
#        plt.axis('off')
#        plt.savefig("matches_iter{:d}.png".format(cnt), bbox_inches='tight')
        ####################
        
        # calculate the centroids
        centroids = feature_centroid(kp1, kp2, matches, source_gray.shape)
                
        if np.sqrt(np.sum((centroids[0] - centroids[1])**2)) <= threshold:
            break
        else:
            # calculate changes of angle
            ref_phi, ref_theta = cubemaps.cartesian2spherical(
                    phi=pose[0], theta=pose[1], 
                    width_grids = np.array([centroids[0][0], centroids[0][0]])/reference.shape[0],
                    height_grids= np.array([centroids[0][1], centroids[0][1]])/reference.shape[1]
                    )
            src_phi, src_theta = cubemaps.cartesian2spherical(
                    phi=pose[0], theta=pose[1], 
                    width_grids = np.array([centroids[1][0], centroids[1][0]])/reference.shape[0],
                    height_grids= np.array([centroids[1][1], centroids[1][1]])/reference.shape[1]
                    )
            delta_phi, delta_theta = ref_phi[0]-src_phi[0], ref_theta[0]-src_theta[0]
            
            # update poses
            phi = pose[0] - UPDATE_RATE*delta_phi
            theta = pose[1] - UPDATE_RATE*delta_theta
            
            phi,theta = correct_angles((phi, theta))
            pose = (np.asscalar(phi), np.asscalar(theta))
            
    # calculate the score of the selected view
    angle = convert_angle(theta, phi, initial_pose)
    if angle is not None:
        score_overlapping = 1 - ((min(len(matches), MIN_OVERLAPPING) - MIN_OVERLAPPING)**2)/(MIN_OVERLAPPING**2)
        score_triangulation = 1 - ((min(angle, MIN_TRIANGULATION) - MIN_TRIANGULATION)**2)/(MIN_TRIANGULATION**2)
        score = score_overlapping + score_triangulation
        
        pose = (pose[1], pose[0]) # convert to (theta, phi)
    else:
        # Fail to find valid views
        source, pose, score = None, None, None
    
    return source, pose, score


def correct_angles( angles: tuple ):
    
    phi   = angles[0]
    theta = angles[1]
    
    if phi < 0:
        phi = phi + 2*np.pi
    elif phi > 2*np.pi:
        phi = phi - 2*np.pi
    if theta < 0:
        theta = -theta
    elif theta > np.pi:
        theta = 2*np.pi - theta
        
    return (phi, theta)
    

def filter_matches(kp1, kp2, matches):
    '''
        It uses RANSAC and fundamental matrix provided by opencv to filter out 
        outliers.
    '''
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_filttered = [ matches[m] for m in np.where(mask[:,0]==1)[0].tolist() ]
    
    return matches_filttered


def feature_centroid(kp1, kp2, matches, image_size):
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
    
    return [ref_centroid, src_centroid]


def convert_angle(theta: float, phi: float, initial_pose: list):
    '''
        It converts delta_theta and delta_phi to the angle between the two 
    '''
    if theta is None or phi is None:
        angle = None
    else:
        phi = phi - initial_pose[0]
        theta = theta - initial_pose[1]
        
        z = np.tan(np.abs(theta))/np.cos(phi)
        h = np.tan(np.abs(phi))
        
        angle = np.asscalar(np.arctan( np.sqrt(z**2 + h**2) ))
    
    return angle


def draw_keypoints(vis, keypoints, color = (255, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 5, color)
    plt.imshow(vis)
    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:36:17 2019

@author: zhantao
"""
import cv2
import os
import numpy as np
import open3d as o3d

from cam360 import Cam360
from spherelib import eu2pol
from cubicmaps import CubicMaps
import matplotlib.pyplot as plt

# thresholds to sort candidate views
MIN_NUM_FEATURE = 20
MIN_OVERLAPPING = 40
MIN_TRIANGULATION = 6*np.pi/180
SAVE_REFERANCE  = 1
SAVE_SOURCE     = 2

UPDATE_RATE = 0.2
    

class sparseMatches:
    '''
    It stores all detected 2D keypoints and the corresponding 3D points. These points
    are written to Point3D.txt and images.txt and are used to initialize the Patch 
    Matching Stereo GPU of the Colmap.
    '''
       
    def __init__(self, keypointFromRef: list, keypointFromSrc: list, keypointsMatches: list):
        
        self.keyPointRef = []
        self.keyPointSrc = [] 
        self.unsavedKeypointsRef = []
        
        for match in keypointsMatches:
            self.keyPointRef.append([keypointFromRef[match.queryIdx].pt[0],
                                     keypointFromRef[match.queryIdx].pt[1]])
            self.keyPointSrc.append([keypointFromSrc[match.trainIdx].pt[0], 
                                     keypointFromSrc[match.trainIdx].pt[1]])  
        self.keyPointRef = np.array(self.keyPointRef).transpose()
        self.keyPointSrc = np.array(self.keyPointSrc).transpose()
        
        self.indexKeypoints = -1
        self.index3Dpoints  = -1
        
    def setIntrinsics(self, camIntrinsics: np.array):
        self.intrinsics = camIntrinsics
        
    def setExtrinsics(self, camRotationMat: np.array, camTranslateionVec: np.array):
        self.rotation = camRotationMat
        self.translation = camTranslateionVec
    
    def triangulateMatches(self):
        projRef = np.eye(3,4)
        projSrc = np.zeros([3,4])
        
        projSrc[0:3, 0:3] = self.rotation
        projSrc[:, 3] = self.translation
        
        projRef = self.intrinsics.dot(projRef)
        projSrc= self.intrinsics.dot(projSrc)
        
        self.point3D = cv2.triangulatePoints(projRef, projSrc, np.float32(self.keyPointRef), np.float32(self.keyPointSrc))
        self.point3D = self.point3D/self.point3D[3,:]
        
        mask = self.point3D[2,:] > 0
        self.point3D = self.point3D[ :, mask]
        self.keyPointRef = self.keyPointRef[:, mask]
        self.keyPointSrc = self.keyPointSrc[:, mask]

         ###### Debug visualization ######
#        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
#        pcd = o3d.geometry.PointCloud()
#        pcd.points = o3d.utility.Vector3dVector(self.point3D.transpose()[:, :3])
#        o3d.visualization.draw_geometries([pcd, mesh_frame])
    
    def saveKeypoint( self, pathToFile: str, RefOrSrc: int, imageIdx: int):
        
        if RefOrSrc == SAVE_REFERANCE and len(self.keyPointRef) > 0:
            keypoints = self.keyPointRef
        elif RefOrSrc == SAVE_SOURCE and len(self.keyPointSrc) > 0:
            keypoints = self.keyPointSrc
        else:
            return None
            
        f = open(pathToFile, "r")
        contents = f.readlines()
        f.close()
        
        checkList = []
        for line in contents:
            checkList.append(str(imageIdx) == line.split(' ')[0])
        index = np.where(checkList)[0][0] + 1
        newline = contents[index][1:] 

        for idx in range(keypoints.shape[1]):
            newline += str(keypoints[0, idx]) + ' ' + str(keypoints[1, idx]) + ' ' + str(self.point3DIdx + idx) + ' '
        contents.insert(index, newline)
            
        f = open(pathToFile, "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()
                        
    def savePoints3D( self, pathToFile: str, refImgIdx: int, srcImgIdx: int ):
        if len(self.point3D) > 0:
            f = open(pathToFile, "r")
            contents = f.readlines()
            f.close()
            
            for index in range(self.point3D.shape[1]):
                newPoint = [ str(index + self.point3DIdx) + ' ' + str(self.point3D[0, index]) + ' ' +  
                             str(self.point3D[1, index]) + ' ' +  str(self.point3D[2, index]) + ' ' + 
                             str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(1) + ' ' + str(refImgIdx) + ' ' +
                             str(index + self.keyPointRefIdx) + ' ' + str(srcImgIdx) + ' ' + str(index + self.keyPointSrcIdx) + '\n']
                
                contents += newPoint
            
            f = open(pathToFile, "w")
            contents = "".join(contents)
            f.write(contents)
            f.close()
            
                            
    def savePoints(self, pathToFile: str, refImgIdx: int, srcImgIdx: int):
       
        point3DFile = os.path.join(pathToFile, 'points3D.txt')
        imagesFile  = os.path.join(pathToFile, 'images.txt')
        
        f = open(point3DFile, "r")
        self.point3DIdx = len(f.readlines())
        f.close()
        
        f = open(imagesFile, "r")
        contents = f.readlines()
        for idx, line in enumerate(contents):
            if str(refImgIdx) == line.split(' ')[0]:
                self.keyPointRefIdx = int((len(contents[idx+1].split(' ')) - 1)/3)
                break
        f.close()
        if line == contents[-1]:    # if the ref view is not saved yet, save its keypoints
            ValueError("By now, the reference image has to be the first image in the list.")
                
        f = open(imagesFile, "r")
        contents = f.readlines()
        for idx, line in enumerate(contents):
            if str(srcImgIdx) == line.split(' ')[0]:
                self.keyPointSrcIdx = int((len(contents[idx+1].split(' ')) - 1)/3)
                break
        f.close()
        
        self.savePoints3D(point3DFile, refImgIdx, srcImgIdx)
        self.saveKeypoint(imagesFile, SAVE_REFERANCE, refImgIdx)
        self.saveKeypoint(imagesFile, SAVE_SOURCE, srcImgIdx)
        

def view_selection(cam:Cam360, reference:np.array, initial_pose: tuple, reference_trans: np.array, fov: tuple=(np.pi/2, np.pi/2),
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
            
        reference_pose : np.array
            Pose of the reference view
            
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
    
    # if reference view and src view is colinear, skip them;
    # theta_ref == theta_src, phi_ref == phi_src or the other side.
    delta_trans = reference_trans - cam.translation_vec
    trans_ref2src = eu2pol(np.array([delta_trans[0]]), np.array([delta_trans[1]]), np.array([delta_trans[2]])) 
    if (abs(trans_ref2src[0] - initial_pose[0]) < 1e-8 and abs(trans_ref2src[1] - initial_pose[1]) < 1e-8) \
    or (abs( np.pi - trans_ref2src[0] - initial_pose[0] ) < 1e-8 and abs( np.mod(trans_ref2src[1] - np.pi, 2*np.pi) - initial_pose[1]) < 1e-8):    
        
        cubemaps = CubicMaps()
        
        pose     = (initial_pose[1], initial_pose[0])
        source   = cubemaps.cube_projection(cam=cam, direction=(pose + fov), resolution=reference.shape)
        score    = 2
        pose     = initial_pose
        Matches = sparseMatches([],[],[])
    
    else:
    
        # initialize objs to be used
        sift = cv2.xfeatures2d.SIFT_create()
        cubemaps = CubicMaps()
        
        # color image to grayscale image
        if reference.max() <= 1:
            reference = np.round(reference*255) 
        reference = cv2.cvtColor(reference.astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        # ATTENTION, here the pose is filipped. [becomes (phi, theta)]
        initial_pose = (initial_pose[1], initial_pose[0])
        
        # image center
        # in opencv, row is y col is x while in numpy row is x col is y
        # so here we flip the two dimensions
        image_center = [reference.shape[1]/2, reference.shape[0]/2]
        
        pose = initial_pose
        centroids  = [image_center, image_center]
        for cnt in range(max_iter):
            
        # Firstly, update theta and phi
            if centroids is not None:
                # calculate changes of angle
                ref_phi, ref_theta = cubemaps.cartesian2spherical(
                        phi=pose[0], theta=pose[1], 
                        width_grids  = np.array([centroids[0][0] - image_center[0]])/(image_center[0]),
                        height_grids = np.array([centroids[0][1] - image_center[1]])/(image_center[1])
                        )
                src_phi, src_theta = cubemaps.cartesian2spherical(
                        phi=pose[0], theta=pose[1], 
                        width_grids  = np.array([centroids[1][0] - image_center[0]])/(image_center[0]),
                        height_grids = np.array([centroids[1][1] - image_center[1]])/(image_center[1])
                        )
                
                delta_phi   = ref_phi[0] - src_phi[0]
                delta_phi   = delta_phi - 2*np.pi*np.sign(delta_phi)*(abs(delta_phi) > np.pi)
                delta_theta = ref_theta[0]-src_theta[0]
                
                # update poses
                phi = pose[0] - UPDATE_RATE*delta_phi
                theta = pose[1] - UPDATE_RATE*delta_theta
                
                phi,theta = correct_angles((phi, theta))
                pose = (np.asscalar(phi), np.asscalar(theta))
            else:
                pose = (np.random.normal(loc=initial_pose[0], scale=0.5),
                        np.random.normal(loc=initial_pose[1], scale=0.25))
                pose = correct_angles(pose)
            
        # Obtain a source view
            source = cubemaps.cube_projection(cam=cam, direction=(pose + fov), resolution=reference.shape)
            source_gray = cv2.cvtColor( np.round(source*255).astype('uint8'), cv2.COLOR_RGB2GRAY )
           
            ##################################
    #        DEMO AND DEBUG
    #        plt.imshow(source)
    #        plt.axis('off')
    #        plt.savefig("view_{:d}.png".format(cnt), bbox_inches='tight')
    #        cubemaps.cube_projection(cam=cam, direction=((3.8,1.57) + fov), resolution=reference.shape)
            ##################################
            
        # feature detection and matching
            kp1, des1 = sift.detectAndCompute(reference, None)
            kp2, des2 = sift.detectAndCompute(source_gray, None)
    
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)
            
            # filtering outliers 
            if use_filter:
                matches = filter_matches(kp1, kp2, matches)
                
            # Can not find enough inlier matches or the selected view looks
            # toward the wrong direction; start random searching
            if len(matches) <= MIN_NUM_FEATURE:
                centroids = None
                continue
            
            ####################            
#            DEMO AND DEBUG
#            show_matches = cv2.drawMatches(reference, kp1, source_gray, kp2, matches, None, flags=2)
#            plt.figure(figsize=(18,14))
#            plt.imshow(show_matches)
#            plt.axis('off')
#            plt.savefig("matches_iter{:d}.png".format(cnt), bbox_inches='tight')
#            plt.close()
#            print("theta: {:f},  phi: {:f}".format(pose[1], pose[0]))
             ####################
            
            # calculate the centroids
            centroids = feature_centroid(kp1, kp2, matches, source_gray.shape)
                    
            if np.sqrt(np.sum((centroids[0] - centroids[1])**2)) <= threshold:      
                break
                
        if centroids is not None:
            # calculate the score of the selected view
            angle = convert_angle(pose[1], pose[0], initial_pose)
            score_overlapping = 1 - ((min(len(matches), MIN_OVERLAPPING) - MIN_OVERLAPPING)**2)/(MIN_OVERLAPPING**2)
            score_triangulation = 1 - ((min(angle, MIN_TRIANGULATION) - MIN_TRIANGULATION)**2)/(MIN_TRIANGULATION**2)
            score = score_overlapping + score_triangulation
            
            pose = (pose[1], pose[0])    # convert to (theta, phi)
            
            # save matches
            Matches = sparseMatches(kp1, kp2, matches)
            
            ####################            
#            DEMO AND DEBUG
#            show_matches = cv2.drawMatches(reference, kp1, source_gray, kp2, matches, None, flags=2)
#            plt.figure(figsize=(18,14))
#            plt.imshow(show_matches)
#            plt.axis('off')
#            plt.savefig("matches_iter{:d}.png".format(cnt), bbox_inches='tight')
#            plt.close()
             ###################
            
        else:
            # Fail to find valid views
            source, pose, score = None, None, None
            Matches = sparseMatches([],[],[])
    
    return source, pose, score, Matches


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
    # reference image center (row -> y in opencv, col -> x in opencv)
    center = (image_size[1]/2, image_size[0]/2)
    
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
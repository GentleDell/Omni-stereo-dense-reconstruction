#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:00:57 2019

@author: zhantao
"""
import cv2
import warnings
import numpy as np
from read_dense import read_array
from spherelib import pol2eu, eu2pol

from typing import Optional
from interpolation.splines import CubicSplines

import cam360

class Depth_tool:
    
    def __init__(self, dist: float = 1, expand_fov: float=1, expand_shape: Optional[float]=1):
        """
            It initializes the object.
                
            Parameters
            ----------    
            dist: float
                distance between the tangent plane and the center of the sphere 
            
            expand_fov : float
                The factor to expand the field of view.
                In order to keep all information during cubic mapping (sphere2cube()),
                this parameters should be set to larger than 1.
                
            expand_shape : float
                The factor to recover onmi image from cube maps. 
                If sphere2cube() is used for cube mapping, this variable will be set automatically. 
                
        """
        self._dist = dist
        self._cubemap = []
        self._depthmap = []
        self._omnimage = None
        self._expand_fov = expand_fov
        self._expand_shape = expand_shape   
        
    @property
    def omnimage(self) -> np.array:
        return self._omnimage
    
    @property
    def cubemap(self) -> list:
        return self._cubemap
    
    @property
    def depthmap(self) -> list:
        return self._depthmap
    
    def save_cubemap(self, prefix: str = ''):
        if len(self._cubemap) == 0:
            warnings.warn("No valid cube map!")
        else:
            for ind, view in enumerate(self._cubemap):
                file_name = prefix + 'view' + str(ind)+'.png'
                cv2.imwrite(file_name, 255*np.flip(view,axis = 2))
    
    def save_cubedepth(self, prefix: str = ''):
        if len(self._depthmap) == 0:
            warnings.warn("No valid cube map!")
        else:
            for ind, view in enumerate(self._depthmap):
                file_name = prefix + 'view' + str(ind)+'.png'
                cv2.imwrite(file_name, 255*(view - np.min(view))/(np.max(view)-np.min(view)))
            
    def save_omnimage(self):
        if self._omnimage is None:
            warnings.warn("No valid omnidirectional image!")
        if self._omnimage.shape[2] == 3: 
            cv2.imwrite("omni_image.png", 255*np.flip(self._omnimage,axis = 2))
        else: 
            cv2.imwrite("omni_depthmap.png", 255*self._omnimage)
    
    
    def cube2sphere( self, cube_list: list = None, normalize: bool = False,
                    resolution: tuple = (256, 512), 
                    order: Optional[tuple] = None):
        """
            It projects a list of six cubic images to an omnidirectional image. The default resolution of the omnidirectional image is 512x1024.
                
            Parameters
            ----------    
            cube_list : a list of 6 cubic images
                If 'position' is not given, then the order of the 6 images has to be:
                [ 0th:  back  |  1st:  left  |  2nd:  front  |  3rd:  right  |  4th:  top  |  5th:  bottom ]
                
            resolution : tuple
                The required resolution for the ouput omnidirectional image, default is 512x1024.
                
            position :
                TBD, support cubicmaps in different orders.
    
            Returns
            -------
            Omni_image : np.array
                An omnidirectional image generated from the given 6 cubic images.
            
            Examples
            --------
            >>> from DepthMap_Tools import Depth_tool
            >>> tool_obj = Depth_tool()
            >>> tool_obj.sphere2cube(Omni_obj)
            >>> Omni_new = tool_obj.cube2sphere( tool_obj._cubemap ) 
        """
        # check the cube list, if it is empty, try to use the depth maps or cubic images of the object
        if cube_list is None:
            if len(self._depthmap) == 6:
                cube_list = self._depthmap.copy()
            elif len(self._cubemap) == 6:
                cube_list = self._cubemap.copy()
            else:
                raise ValueError('Bad input! Invalid input image list.')       
        if normalize:
            for ct in range(len(cube_list)):
                cube_list[ct] = (cube_list[ct] - np.min(cube_list[ct]))/(np.max(cube_list[ct]) - np.min(cube_list[ct]))
                
        if self._expand_fov == 1 or self._expand_shape == 1:
            warnings.warn("The omnidirectional image could have some streaks as '_expand_fov' is using the default value.") 
        
        # check the consistency of images' size
        if cube_list[0].shape != cube_list[1].shape or cube_list[1].shape != cube_list[2].shape or  \
           cube_list[2].shape != cube_list[3].shape or cube_list[3].shape != cube_list[4].shape or  \
           cube_list[4].shape != cube_list[5].shape:
            print('Bad input! All given images should have the same size.')
            return None            
        
        # obtain input image size
        width  = cube_list[0].shape[1]
        height = cube_list[0].shape[0] 
        channel= cube_list[0].shape[2] 
        
        # parameters for cubicsplines
        up  = [2*self._expand_shape, 2*self._expand_shape]
        low = [-self._expand_shape*2/width, -self._expand_shape*2/width]
        orders = [height, width]
        
        # create the Omnidirectional image
        Omni_image = np.zeros([resolution[0]*resolution[1], channel])
        
        # if the order of images is given 
        if order is not None:
            print('Setting different order is not supported now.')
            # TODO:
                # support different orders
        
        # use the default image order 
        else:
            for face in range(len(cube_list)):
                # Create a spline approximation of the camera texture.
                spline = CubicSplines(low, up, orders, 
                                      np.reshape(cube_list[face], (height * width, channel)))
                
                points, mask_face  = self.spherical2img(face, resolution)
                Omni_image[ mask_face, :]  = spline.interpolate(points, diff=False)
                
            self._omnimage = None
            self._omnimage = Omni_image.reshape([resolution[0], resolution[1], -1])
            
    
    
    def spherical2img(self, face_num: int, resolution: np.array) -> np.array:
        """
            For each pixel on the unit sphere, this function computes the position of the pixel in corresponding cubic image.
                
            Parameters
            ----------    
            face_num : int
                The face number. [ 0:  back  |  1:  left  |  2:  front  |  3:  right  |  4:  top  |  5:  bottom ]
            resolution : tuple
                The required resolution for theta and phi.
                
            Returns
            -------
            points : numpy.array 
                The positions (under image coordinate) of the points on sphere, with refer to the face_num.
            mask_face : numpy.array
                A mask vector for the image block on omnidirectional image corresponding to the face_num.
        """
        # generate sphere grids
        phi   = np.linspace(0, 2*np.pi, num = resolution[1])
        theta = np.linspace(0, np.pi  , num = resolution[0])
        grid_phi, grid_theta = np.meshgrid(phi, theta)
        
        # deal with the 4 horizontal surfaces
        if face_num < 4 and face_num >= 0:
            # generate mask using element wise bool operation
            if face_num == 0:
                mask_phi = (grid_phi - face_num*np.pi/2 < np.pi/4) + (grid_phi - face_num*np.pi/2 > -np.pi/4 + (face_num==0)*2*np.pi)
            else:
                mask_phi = (grid_phi - face_num*np.pi/2 < np.pi/4) * (grid_phi - face_num*np.pi/2 > -np.pi/4 + (face_num==0)*2*np.pi)
            mask_theta = (grid_theta > np.pi/4) * (grid_theta < 3*np.pi/4)
            mask_face  = mask_theta * mask_phi
            mask_face  = mask_face.flatten()   
            
            # slicing grids
            phi   = -( grid_phi.flatten()[mask_face] - face_num*np.pi/2 )
            theta = grid_theta.flatten()[mask_face]
            
            # normalized image coordinates
            u = self._expand_shape - np.tan(phi)
            v = self._expand_shape - np.sqrt(1 + np.power(np.tan(phi), 2))/np.tan(theta)
        
        # the top surface 
        elif face_num == 4:
            mask_face = (grid_theta < np.pi/4)
            mask_face = mask_face.flatten()
            
            phi   = grid_phi.flatten()[mask_face]
            theta = grid_theta.flatten()[mask_face]
            
            # normalized image coordinates
            u = self._expand_shape - np.tan(theta)*np.sin(phi)
            v = self._expand_shape - np.tan(theta)*np.cos(phi)
        
        # the bottom surface       
        elif face_num == 5:
            mask_face = (grid_theta > 3*np.pi/4)
            mask_face = mask_face.flatten()
            
            phi   = grid_phi.flatten()[mask_face]
            theta = grid_theta.flatten()[mask_face]
            
            # normalized image coordinates
            u = self._expand_shape + np.tan(theta)*np.sin(phi)
            v = self._expand_shape - np.tan(theta)*np.cos(phi)
        else:
            raise ValueError('Bad input! The give surface number is not supported.')
            
        points  = np.column_stack((v, u))
        
        return points, mask_face
    
    
    
    def sphere2cube (self, cam:'cam360', resolution: tuple=(256,256), dist: float=None):
        """
            Description: 
            ----------
            It generates cubic maps from the given omnidirectional image.    
            
            Parameters
            ----------    
            cam: object
                camera320 object to be projected
            
            resolution: tuple
                resolution of the cubic maps
                
            dist: float
                distance between the tangent plane and the center of the sphere 
                
            Examples
            --------
            >>>> sphere2cube(Omni_obj, rsesolution=(256,256))
        """
    # input verification
        if cam.texture is None:
            raise ValueError('INPUT ERROR! Empty cam360 object.')  
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError('INPUT ERROR! Resolutions must be positive.')  
        if len(self._cubemap) != 0:
            self._cubemap = []
        # if dist is not given, use the default value    
        if dist is None:
            dist = self._dist
    # generate cubic maps
        # enlarge field of view to include more information for further cube->sphere reconstruction4
        if self._expand_fov == 1:
            warnings.warn(" '_expand_fov' is set to the default value.\nIt could lead to information loss during the sphere->cube projection and cube->sphere projection\nInitialize it properly with 1.25 or larger is strongly recommended.") 
        delta_angle = self._expand_fov*np.pi/2
        self._expand_shape = dist*np.tan(delta_angle/2)
        
        # back view
        self._cubemap.append(self.cube_projection(cam, (0,         np.pi/2, delta_angle, delta_angle), resolution))
        # left view 
        self._cubemap.append(self.cube_projection(cam, (np.pi/2,   np.pi/2, delta_angle, delta_angle), resolution))
        # front view
        self._cubemap.append(self.cube_projection(cam, (np.pi,     np.pi/2, delta_angle, delta_angle), resolution))
        # right view
        self._cubemap.append(self.cube_projection(cam, (3*np.pi/2, np.pi/2, delta_angle, delta_angle), resolution))
        # bottom view - from bottom to top
        self._cubemap.append(self.cube_projection(cam, (np.pi,     0,       delta_angle, delta_angle), resolution))
        # top view - from top to bottom
        self._cubemap.append(self.cube_projection(cam, (np.pi,     np.pi,   delta_angle, delta_angle), resolution))
        
    
    def cube_projection( self, cam:'cam360', direction: tuple,
                         resolution: tuple=(256,256), dist: float = None ) -> np.array:
        """
            Description: 
            ----------
            It projects a omnidirectional image to its tangent plane(or the plane parallel to its tangent plane) on the given direction.    
            
            Parameters
            ----------    
            cam: object
                camera320 object to be projected
                 
            direction: tuple (phi, theta, fov_phi, fov_theta)
                phi, theta -> the direction of normal line to the tangent plan.
                fov_phi, fov_theta -> field of view on phi and theta
                
            resolution: tuple
                resolution of the output image
                
            dist: float
                distance between the tangent plane and the center of the sphere 
                
            Returns
            -------
            projection: np.array
                the projected cubic map of the given omnidirectional image on the given plane
                
            Examples
            --------
            >>> cube_projection(Omni_obj, direction=(np.pi, np.pi/2, np.pi/2, np.pi/2), resolution=(320,320))
        """
    # input verification
        if len(direction)!=2 and len(direction)!=4:
            raise ValueError('INPUT ERROR! the input direction must contain 2 angles as (phi, theta) or contain 4 angles as (phi, theta, fov_phi, fov_theta)')   
        elif len(direction)==2:
            center_phi   = direction[0]
            center_theta = direction[1]
            fov_phi   = np.pi/2
            fov_theta = np.pi/2
        else:
            center_phi   = direction[0]
            center_theta = direction[1]
            fov_phi   = direction[2]
            fov_theta = direction[3]
            
        # if dist is not given, use the default value    
        if dist is None:
            dist = self._dist
        elif dist < 1:
            raise ValueError('INPUT ERROR! The distance between the tangent plane and the center of the sphere should be greater than 1')   
        
        if center_phi < 0 or center_phi >= 2*np.pi :
            raise ValueError('INPUT ERROR! Phi of the normal line should belong to [0, 2*pi)')   
        if center_theta < 0 or center_theta > np.pi:
            raise ValueError('INPUT ERROR! theta of the normal line should belong to [0, pi]')   
        if fov_phi < 0 or fov_phi >= np.pi:
            raise ValueError('INPUT ERROR! fov_phi should belong to [0, pi)')   
        if fov_theta < 0 or fov_theta >= np.pi:
            raise ValueError('INPUT ERROR! fov_theta should belong to [0, pi)') 
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError('INPUT ERROR! Resolutions must be positive.')  
    
    # generate pixel grids    
        # comput the relative image size
        half_height= dist*np.tan(fov_theta/2)        
        half_width = dist*np.tan(fov_phi/2)
        # compute pixels' positions on the image
        width_grids = np.linspace(-half_width , half_width , num=resolution[0], endpoint=True)
        height_grids= np.linspace(-half_height, half_height, num=resolution[1], endpoint=True)
        # generate pixels grids
        width_grids, height_grids = np.meshgrid(width_grids,height_grids)
        
    # compute the corresponding spherical coordinate for every pixel
        phi_range, theta_range = self.cartesian2spherical(center_phi, center_theta, width_grids, height_grids, dist)
    
    # obtain the texture     
        # flatten theta and phi to use the get_texture_at() method
        phi   = phi_range.flatten()
        theta = theta_range.flatten()        
        # get the texture and save it to the output cubemap list
        texture_vec = cam.get_texture_at(theta, phi)
        projection  = texture_vec.T.reshape( resolution[1], resolution[0], 3)
            
        return projection


    def cartesian2spherical( self, phi:float, theta:float, 
                             width_grids: np.array, height_grids: np.array, dist: float = None) -> tuple:
        """
            Description:
            ----------
            It computes the spherical coordinates of the given pixel grids.
                
            Parameters
            ----------    
            phi: float
                the horizontal angle (from negative y axis) of the normal line
                
            theta: float
                the vertical angle (from positive z axis) of the normal line
                
            width_grids: np.array
                positions of points on the image plane. 
                
            height_grids: np.array
                positions of points on the image plane. 
                
            dist: float
                distance between the tangent plane and the center of the sphere 
                
                         width
              z ^        ------                 
                |   y    \--\--\ heigh
                |   /     \--\--\
                |  /       \--\--\
                | /
                |/
                --------------> x
            
            Returns
            -------
            phi_grids: np.array
                phi positions of points under the polar coordinate system
                
            theta_grids: np.array
                theta positions of points under the polar coordinate system
                
            Examples
            --------
            >>> cartesian2spherical(center_phi, center_theta, width_grids, height_grids, dist)
        """  
        # if dist is not given, use the default value    
        if dist is None:
            dist = self._dist
            
        # get the euler coordinate of the tangent point
        center_x, center_y, center_z = pol2eu(theta, phi, dist)
        # compute the length of the projected normal line segment
        prj_dist = dist*np.sin(theta)
        # compute z-axis positions of each point
        zeta_range = -height_grids*np.sin(theta) + center_z
        
        # compute the cooresponding polar coordinate for each image point
        theta_grids = np.arccos(zeta_range/np.sqrt(dist**2 + height_grids**2 + width_grids**2))
        phi_grids   = np.arctan2(width_grids,  prj_dist+height_grids*np.cos(theta)) + phi
        
        # make sure every angle is valid phi: (0,2pi), theta:(0,pi)
        phi_grids[phi_grids<0] = phi_grids[phi_grids<0] + 2*np.pi
        phi_grids[phi_grids>=2*np.pi] = phi_grids[phi_grids>=2*np.pi] - 2*np.pi
        
        theta_grids[theta_grids<0] = np.abs(theta_grids[theta_grids<0])
        theta_grids[theta_grids>np.pi] = 2*np.pi - theta_grids[theta_grids>np.pi]
        
        return phi_grids, theta_grids
    
    
    def load_depthmap(self, path_to_file: list):
        if len(path_to_file)==6:
            if len(self._depthmap) != 0:
                warnings.warn("Depth maps will be replaced!")
                self._depthmap = []
            for ct in range(6):
                raw_depthmap = read_array(path_to_file[ct])
                depth_map = self.filt_depthoutliers(raw_depthmap)
                self._depthmap.append(np.expand_dims(depth_map, axis = 2))
                
        elif len(path_to_file)==4:
            if len(self._depthmap) != 0:
                warnings.warn("Depth maps will be replaced!")
                print('Inputs are treated as back, left, front and righ view')
                self._depthmap = []
            
            for ct in range(4):
                raw_depthmap = read_array(path_to_file[ct])
                depth_map = self.filt_depthoutliers(raw_depthmap)
                self._depthmap.append(depth_map)
            self._depthmap.insert(0, np.zeros(self._depthmap[1].shape))
            self._depthmap.append(np.zeros(self._depthmap[1].shape))
        else :
            raise ValueError('Bad input! Only support 4 and 6 depthmaps.')
            
            
    def filt_depthoutliers(self, depth_map: np.array) -> np.array:
        min_depth, max_depth = np.percentile(depth_map, [5, 95])
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        depth_map[depth_map < 0] = 0
        return depth_map
    
    
#    def generate_cameramodel(rotation: list, translation: list, prefix: str):
        
        
        
        
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:00:57 2019

@author: zhantao
"""
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from interpolation.splines import CubicSplines

import cam360

class Depth_tool:
    
    def __init__(self, expand_ratio: float = 1.5):
        
        self._cubemap = []
        
        self._expand_ratio = expand_ratio
    
    
    def get_partial_img (self, cam: 'cam360', theta: np.array, phi: np.array) -> np.array:
        """
            This is a wrapper of cam360's 'get_texture_at()'. It computes the texture at the specified (theta, phi) coordinate range.
    
            Args:
                theta: the target elevation coordinate (M,).
                phi: the target azimuth coordinate (N,).
    
            Returns:
                texture: texture (N, M, self._channels) at the input coordinate.
                
            Example:
                theta = array[1,2,3]
                phi   = array[4,5,6]
                
                return: image data at [1,4], [1,5], [1,6], ... , [3,5], [3,6] and being reshaped back as an image
        """
        
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        phi_grid, theta_grid = phi_grid.flatten(), theta_grid.flatten()
        
        partial_img = cam.get_texture_at(theta_grid, phi_grid)
        partial_img = partial_img.T.reshape( phi.shape[0], theta.shape[0], 3)
        
        return partial_img
    
    
    def cube2sphere( self, cube_list: list = None, resolution: np.array = np.array([512, 1024]), position: Optional[np.array] = None) -> np.array:
        """
            It projects a list of six cubic images to an omnidirectional image. The default resolution of the omnidirectional image is 512x1024.
                
            Parameters
            ----------    
            cube_list : a list of 6 cubic images
                If 'position' is not given, then the order of the 6 images has to be:
                [ 0th:  back  |  1st:  left  |  2nd:  front  |  3rd:  right  |  4th:  top  |  5th:  bottom ]
            resolution : np.array
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
        # check the cube list, if it is empty, try to use the cubic images contained in the object
        if cube_list is None:
            if self._cubemap is not None:
                cube_list = self._cubemap
            else:
                raise ValueError('Bad input! Unvalid input image list.')            
        
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
        up  = [2*self._expand_ratio, 2*self._expand_ratio]
        low = [-self._expand_ratio*2/width, -self._expand_ratio*2/width]
        orders = [height, width]
        
        # create the Omnidirectional image
        Omni_image = np.zeros([resolution[0]*resolution[1], 3])
        
        # if the order of images is given 
        if position:
            print('new order')
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
                
            Omni_image = Omni_image.reshape([resolution[0], resolution[1], 3])
            
        return Omni_image
    
    
    def spherical2img(self, face_num: int, resolution: np.array) -> np.array:
        """
            For each pixel on the unit sphere, this function computes the position of the pixel in corresponding cubic image.
                
            Parameters
            ----------    
            face_num : int
                The face number. [ 0:  back  |  1:  left  |  2:  front  |  3:  right  |  4:  top  |  5:  bottom ]
            resolution : np.array
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
            u = self._expand_ratio - np.tan(phi)
            v = self._expand_ratio - np.sqrt(1 + np.power(np.tan(phi), 2))/np.tan(theta)
        
        # the top surface 
        elif face_num == 4:
            mask_face = (grid_theta < np.pi/4)
            mask_face = mask_face.flatten()
            
            phi   = grid_phi.flatten()[mask_face]
            theta = grid_theta.flatten()[mask_face]
            
            # normalized image coordinates
            u = self._expand_ratio - np.tan(theta)*np.sin(phi)
            v = self._expand_ratio - np.tan(theta)*np.cos(phi)
        
        # the bottom surface       
        elif face_num == 5:
            mask_face = (grid_theta > 3*np.pi/4)
            mask_face = mask_face.flatten()
            
            phi   = grid_phi.flatten()[mask_face]
            theta = grid_theta.flatten()[mask_face]
            
            # normalized image coordinates
            u = self._expand_ratio + np.tan(theta)*np.sin(phi)
            v = self._expand_ratio - np.tan(theta)*np.cos(phi)
        else:
            raise ValueError('Bad input! The give surface number is not supported.')
            
        points  = np.column_stack((v, u))
        
        return points, mask_face
        
    
    def sphere2cube( self, cam: 'cam360', resolution: int = 512) -> list:
        """
            It computes the six cubic maps of the given cam360 object. The default resolution is 512x512.
                
            Parameters
            ----------    
            cam : cam360 object
                containing a valid omnidirectional image.
            resolution : int
                the required resolution of cubic maps, default is 512x512.
    
            Returns
            -------
            cubemap : a list of the six cubic maps of the given cam360 object.
                      [ 0th:  back  |  1st:  left  |  2nd:  front  |  3rd:  right  |  4th:  top  |  5th:  bottom ]
                
            Examples
            --------
            >>> from DepthMap_Tools import Depth_tool
            >>> tool_obj = Depth_tool()
            >>> cubemaps = tool_obj.sphere2cube(Omni_obj, resolution=1024)
        """
        
        if cam._texture is None:
            print("Unvalid input cam360 Object!")
            return None
        
        # if the cubmap is not 
        if len(self._cubemap) != 0:
            self._cubemap = []
        
        # the cartesian coordinate, format: [6 maps, resolution , resolution , channels]
        xyz_space = np.zeros((6, resolution, resolution, 3))        
        
        # generate an all-one vector and the 2D mesh grid
        Ones = np.ones((resolution,resolution,1))
        
        cube_coor = np.linspace(-cam._radius * self._expand_ratio, cam._radius * self._expand_ratio, num = resolution)
        cube_grid_x, cube_grid_y = np.meshgrid(cube_coor,cube_coor)
        cube_grid_x, cube_grid_y = np.expand_dims(cube_grid_x,axis = -1), np.expand_dims(cube_grid_y,axis = -1)
        
        # generate the cubic grid, each point of the grid corresponds to a pixel of the six cubic maps
        # format: 0th-faces, 1st-width(resolution), 2nd-height(resolution), 3rd-[x,y,z]
        xyz_space[0,:,:,:] = np.dstack( (-cube_grid_x ,  -Ones      , -cube_grid_y))
        xyz_space[1,:,:,:] = np.dstack( (   -Ones     , cube_grid_x , -cube_grid_y))
        xyz_space[2,:,:,:] = np.dstack( ( cube_grid_x ,   Ones      , -cube_grid_y))
        xyz_space[3,:,:,:] = np.dstack( (    Ones     ,-cube_grid_x , -cube_grid_y))
        xyz_space[4,:,:,:] = np.dstack( ( cube_grid_x , cube_grid_y ,     Ones    ))
        xyz_space[5,:,:,:] = np.dstack( ( cube_grid_x ,-cube_grid_y ,    -Ones    ))
        
        # convert cartesian coordinate to spherical coordinate
        sph_space = self.cartesian2spherical(xyz_space)
        
        # get the texture of each cubic map        
        for angles in sph_space:
            
            # flatten theta and phi to use the get_texture_at() method
            theta = angles[:,:,0].flatten()
            phi   = angles[:,:,1].flatten()
            
            # get the texture and save it to the output cubemap list
            texture_vec = cam.get_texture_at(theta, phi)
            self._cubemap.append(texture_vec.T.reshape( resolution, resolution, 3))
            
        return self._cubemap
        
    
    def cartesian2spherical( self, xyz_coordinates: np.array ) -> np.array:
        """
            It computes the spherical coordinates of the given points, using the their cartesian coordinates.
                
            Parameters
            ----------    
            xyz_coordinates : numpy.array
                The cartesian coordinate of each point. 
                It contains 4 dimensions:  0th-faces, 1st-width(resolution), 2nd-height(resolution), 3rd-[x,y,z]
    
            Returns
            -------
            sphere_coord : `list` of numpy.array 
                The number of entries equals to xyz_coordinates.shape[0], i.e. the number of faces). Each entry of sphere_coord is a numpy.array.
                Each array contains 3 dimensions:  0th-width(resolution), 1st-height(resolution), 2nd-[theta, phi]
                
            Examples
            --------
            >>> from DepthMap_Tools import Depth_tool
            >>> cat_space = np.ones((2, 10, 10, 3))
            >>> tool_obj  = Depth_tool()
            >>> sph_space = tool_obj.cartesian2spherical(cat_space)
        """
        # the output list, containing spherical coordinates corresponding to the given cartesian coordinates
        sphere_coord = []; 
        
        for face in range(xyz_coordinates.shape[0]):
            # theta
            theta = np.arctan2( np.sqrt(xyz_coordinates[face,:,:,0]**2 + xyz_coordinates[face,:,:,1]**2), xyz_coordinates[face,:,:,2] )
            
            # phi
            phi   = np.arctan2( xyz_coordinates[face,:,:,1], xyz_coordinates[face,:,:,0])
            
            # align two coordinate system (ours: neg-Y [0, 2*pi] , numpy: pos-X [-pi, pi] )
            mask_neg = phi < 0
            mask_pos = phi > 0
            phi[mask_neg]  = -(phi[mask_neg] + np.pi/2)
            phi[mask_pos]  = - phi[mask_pos] + np.pi*3/2 
            
            # avoid being out of the domain of phi [0, 2*pi]
            mask_neg = phi < 0
            phi[mask_neg]  = phi[mask_neg] + 2*np.pi

            # stack theta and phi
            theta_phi = np.dstack( (np.expand_dims(theta, axis=-1), np.expand_dims(phi, axis=-1)) )
            
            sphere_coord.append(theta_phi)
        
        return sphere_coord
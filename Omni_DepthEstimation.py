# add the path to the necessary files to the sys path temporarily
import sys
sys.path.append('./cube_script')

import matplotlib.pyplot as plt
from workspace_helper import create_workspace, create_depth_workspace, project_colmap_depth, evaluate
# In[]
Camera_parameter = [256,256, 256,256]

# In[]
create_workspace(image_dir='../../dataset/omnidirectional/classroome_nochair_1024_512', 
                 file_suffix='png', 
                 work_dir = '../Visualization/work_space/classroom/1024_512', 
                 camera_parameters=[Camera_parameter], reference_pose=[1,0,0,0,0,0,1],
                 resolution=(512,512))


# In[]
#create_depth_workspace(image_dir='../../../dataset/omnidirectional/512_256/allviews', file_suffix='exr', 
#                       work_dir = '../../Vis_Data/depthmap/classroom/GroundTruth', 
#                       cam_para = Camera_parameter, resolution=(256,256))


# In[]
'''Depth map parts'''
#import cv2
#
#path_to_dmap = '../../Vis_Data/depthmap/castle/fixed/allviews'
#name_pattern = 'test_0_0_1_1024_5120000'
#
#Camera_parameter = [256,256,256,256]
#estimated = project_colmap_depth(path = path_to_dmap, 
#                                view_name = name_pattern,
#                                views_list = [0,1,2,3],
#                                output_resolution=(512, 1024), 
#                                use_radial_dist=True, 
#                                camera_para=Camera_parameter,
#                                save=False)
#
#GT = cv2.imread('../../../dataset/omnidirectional/1024_512/allviews/test_0_0_1_1024_512_depth0000.exr',cv2.IMREAD_ANYDEPTH)
#GT[GT>100] = 100
#
#evaluate(estimation=estimated[:,:,0], GT=GT, checking_line=255, save=False)
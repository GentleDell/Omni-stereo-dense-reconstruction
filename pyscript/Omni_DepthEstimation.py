import matplotlib.pyplot as plt
from workspace_helper import create_workspace, create_depth_workspace, project_colmap_depth, evaluate

#create_workspace(image_dir='../../../dataset/omnidirectional/512_256/multiviews', file_suffix='png', work_dir = '../work_space/classroom/multiviews', 
#                 camera_parameters=[[268.05641768, 268.05641768, 128,128]], reference_pose=[1,0,0,0,0,1,1],
#                 resolution=(256,256))
#
#create_depth_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='exr', work_dir = '../data', 
#                       cam_para = [267,267,256,256])


'''Depth map parts'''
path_to_dmap = '../../Vis_Data/depthmap/castle/fixed/allviews'
name_pattern = 'test_4_0_1_1024_5120000'

estimated = project_colmap_depth(path = path_to_dmap, 
                                 view_name = name_pattern,
                                 views_list = [0,1,2,3],
                                 output_resolution=(512, 1024), 
                                 use_radial_dist=True, 
                                 camera_para=[267,267,256,256],
                                 save=False)
import cv2
GT = cv2.imread('../../../dataset/omnidirectional/1024_512/allviews/test_0_0_1_1024_512_depth0000.exr',cv2.IMREAD_ANYDEPTH)
GT[GT>100] = 100

evaluate(estimation=estimated.squeeze(axis=2), GT=GT, checking_line=255, save=False)
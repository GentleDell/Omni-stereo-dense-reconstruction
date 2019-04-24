import numpy as np
from colmaplib import create_workspace, create_depth_workspace

create_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='png', work_dir = '../data', 
                 camera_parameters=[[267,267,256,256]], reference_pose=np.array([1,0,0,0,0,0,1]))
#
#create_depth_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='exr', work_dir = '../data', 
#                       cam_para = [267,267,256,256])

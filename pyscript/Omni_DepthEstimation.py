import matplotlib.pyplot as plt
from workspace_helper import create_workspace, create_depth_workspace, project_colmap_depth

create_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='png', work_dir = '../data', 
                 camera_parameters=[[267,267,256,256]], reference_pose=[1,0,0,0,0,0,1])

create_depth_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='exr', work_dir = '../data', 
                       cam_para = [267,267,256,256])

path_to_dmap = ['../depthmap/fixed/allviews/view0/test_0_0_1_1024_5120000_view0.png.geometric.bin',
                '../depthmap/fixed/allviews/view1/test_0_0_1_1024_5120000_view1.png.geometric.bin',
                '../depthmap/fixed/allviews/view2/test_0_0_1_1024_5120000_view2.png.geometric.bin',
                '../depthmap/fixed/allviews/view3/test_0_0_1_1024_5120000_view3.png.geometric.bin']
#                '../depthmap/fixed/allviews/view4/test_0_0_1_1024_5120000_view4.png.geometric.bin',
#                '../depthmap/fixed/allviews/view5/test_0_0_1_1024_5120000_view5.png.geometric.bin']

estimated = project_colmap_depth(path_to_file=path_to_dmap, output_resolution=(512, 1024), 
                                 use_radial_dist=True, camera_para=[267,267,256,256],
                                 save=False)
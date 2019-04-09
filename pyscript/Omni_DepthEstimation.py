import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

from cam360 import Cam360
from DepthMap_Tools import Depth_tool

filepath = '../../../dataset/omnidirectional/1024_512/'
filename = 'test_-4_-4_1_1024_5120000.png'
Omni_img = cv2.imread(filepath + filename)
#Omni_img = cv2.imread('../../../dataset/omnidirectional/512_256/test_0_0_1_512_256_0111.png')

# as opencv use BGR channel of images, here the channel order has been flipped
Omni_img = np.flip(Omni_img, axis=2)
Omni_img = Omni_img/np.max(Omni_img)

# initial parameters for the omnidirectional image
height   = Omni_img.shape[0]
width    = Omni_img.shape[1]
channels = Omni_img.shape[2]
rotation_mtx     = np.eye(3)
translation_vec  = np.zeros([3,1])

# Initialize Omnidirectional object
tool_obj = Depth_tool(expand_fov=1.25)
Omni_obj = Cam360(rotation_mtx, translation_vec, height, width, channels, Omni_img)

tool_obj.sphere2cube(Omni_obj, resolution=(512,512))
tool_obj.cube2sphere( resolution=(512,1024) )
#tool_obj.save_omnimage()
#tool_obj.save_cubemap(prefix=filename[:-8] + 'maps')

depth_path=['../depthmap/depthmapGPU/pose_multiview0/test_0_0_1_1024_5120000.pngview0.png.geometric.bin',
            '../depthmap/depthmapGPU/pose_multiview1/test_0_0_1_1024_5120000.pngview1.png.geometric.bin',
            '../depthmap/depthmapGPU/pose_multiview2/test_0_0_1_1024_5120000.pngview2.png.geometric.bin',
            '../depthmap/depthmapGPU/pose_multiview3/test_0_0_1_1024_5120000.pngview3.png.geometric.bin',
            '../depthmap/depthmapGPU/pose_multiview4/test_0_0_1_1024_5120000.pngview4.png.geometric.bin',
            '../depthmap/depthmapGPU/pose_multiview5/test_0_0_1_1024_5120000.pngview5.png.geometric.bin',]
tool_obj.load_depthmap(depth_path)
tool_obj.cube2sphere(normalize=True, resolution=(512,1024))
#tool_obj.save_omnimage()
#tool_obj.save_cubedepth(prefix=filename[:-8] + 'depth')


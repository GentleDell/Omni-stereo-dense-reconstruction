import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

from cam360 import Cam360
from DepthMap_Tools import Depth_tool

#Omni_img = cv2.imread('../../dataset/omnidirectional/EPFL_MMSPG_Omni/SDR version/Indoor/IMG009/Indoor_IMG009_mantiuk08.jpg')
Omni_img = cv2.imread('../../dataset/omnidirectional/1024_512/test_0_-4_1_1024_5120000.png')

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
tool_obj = Depth_tool(expand_ratio=1.36)
Omni_obj = Cam360(rotation_mtx, translation_vec, height, width, channels, Omni_img)

tool_obj.sphere2cube(Omni_obj, resolution=256)
tool_obj.save_cubemap()
Omni_new = tool_obj.cube2sphere( resolution=np.array([256,512])) 


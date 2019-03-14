import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam360 import Cam360
from DepthMap_Tools import Depth_tool

#Omni_img = cv2.imread('../../dataset/omnidirectional/EPFL_MMSPG_Omni/SDR version/Indoor/IMG009/Indoor_IMG009_mantiuk08.jpg')
Omni_img = cv2.imread('../../dataset/omnidirectional/Virtual_scenes/512_256/test_0_0_1_512_256_0111.png')
Omni_img = Omni_img/np.max(Omni_img)

height   = Omni_img.shape[0]
width    = Omni_img.shape[1]
channels = Omni_img.shape[2]
rotation_mtx     = np.eye(3)
translation_vec  = np.zeros([3,1])

# Initialize Omnidirectional object
tool_obj = Depth_tool(expand_ratio=1.5)
Omni_obj = Cam360(rotation_mtx, translation_vec, height, width, channels, Omni_img)

tool_obj.sphere2cube(Omni_obj, resolution=256)
Omni_new = tool_obj.cube2sphere( cube_list=tool_obj._cubemap, resolution=np.array([256,512]) ) 


#plt.imshow(tool_obj._cubemap[0])
#plt.savefig("0backward.png", dpi = 300)
#plt.imshow(tool_obj._cubemap[1])
#plt.savefig("1leftward.png", dpi = 300)
#plt.imshow(tool_obj._cubemap[2])
#plt.savefig("2forward.png", dpi = 300)
#plt.imshow(tool_obj._cubemap[3])
#plt.savefig("3rightward.png", dpi = 300)
#plt.imshow(tool_obj._cubemap[4])
#plt.savefig("4upward.png" , dpi = 300)
#plt.imshow(tool_obj._cubemap[5])
#plt.savefig("5downward.png", dpi = 300)
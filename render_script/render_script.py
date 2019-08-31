
"""
Given a blender file "file_name.blender", this script permits to place a 360 degree cameras with arbitrary pose in the scene and get the picture.

To launch the rendering:
    1. Edit the render.cfg file (next to this scrip). If there is no render.cfg then create one and put it in the same folder as this script. You can configure the pose of the 360 camera, resolution of output image and in which GPU you want to render the image. An example is provided below:
            
            # There are 4 gpus in total and we want to use gpu0 to render the 
            # scene. The camera is placed at (X=10, Y=30, Z=1) and rotated by 
            # 90 degree around the X-axis. Resolution is (1024,512)
            all_gpu 4
            gpu 0
            poses 90 0 0 10 30 1
            res 1024 512
            
    2. If blender has been downloaded to or installed in your system, then open a terminal, enter the folder containing this script ("render.py"), and run the following command:
            
        /path/to/blender_folder/blender -b scene_name.blender -P render.py
        
    3. If gpu index is not given, the test_GPU.py can be used to output gpu information by a similar command: (only support blender 2.79)

        /path/to/blender_folder/blender -b scene_name.blender -P test_GPU.py

The images will be saved in the folder "outputImages", within the same folder of this script.

"""

import bpy
import os
import numpy as np

# In[]

def read_cofig():
    '''
        read configuration file
        format of render.cfg:
           '#'    : followed by comments
           'gpu'  : followed by gpu_index
           'pose' : followed by pose data: rotation (x-y-z euler in degree) + position
           'res'  : resolution
           'all_gpu': followed by the number of gpus.
    '''
    poses = None
    all_gpu = None
    gpu_index = None
    resolution = None
    
    
    cfg_file = 'render.cfg'
    
    try:
        os.path.isfile(cfg_file)
    except:
        raise ValueError('No valid configuration file found')
    
    with open(cfg_file, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            else:
                if line[:3].lower() == 'gpu':
                    gpu_index = int(line[4])
                    
                elif line[:5].lower() == 'poses':
                    poses = [ float(item) for item in line[5:].split(' ') if len(item) > 0]
                    
                elif line[:3].lower() == 'res':
                    resolution = [ int(item) for item in line[3:].split(' ') if len(item) > 0 ]
                    
                elif line[:7].lower() == 'all_gpu':
                    all_gpu = int(line[4])
        
    if poses is None:
        raise ValueError('No valid poses')
        
    return all_gpu, gpu_index, poses, resolution


# In[]
# Render

num_all_gpu, gpu_ind, poses, resolution = read_cofig()

# if GPU is enabled, check whether it is available
if (gpu_ind is not None) and (num_all_gpu is not None):
    try:
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    except:
        raise ValueError('Can not find valid CUDA version')
        
    for i in range(num_all_gpu+1):
        bpy.context.user_preferences.addons['cycles'].preferences.devices[i].use = gpu_ind == i


# Create a new camera.
bpy.ops.object.camera_add()

# After a call to bpy.ops.object.primitive_add(), the created object is assigned to the variable bpy.context.object.
cam = bpy.context.object

# Rename the new camera (not necessary).
cam.name = 'Camera_360'

# Turn the camera into an omnidirectional one.
cam.data.type = 'PANO'
cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'

##############################################################################################################################
######################################################### MODIFY HERE ########################################################
##############################################################################################################################

# Camera resolution (e.g., (720, 480), (1920, 1080)).
resolutions = [tuple(resolution)]

# Camera positions in the scene.
camera_coordinates = [tuple(poses[3:])]

# Define the camera field of view.
cam.data.cycles.latitude_min = -np.pi/2
cam.data.cycles.latitude_max = np.pi/2
cam.data.cycles.longitude_min = -np.pi
cam.data.cycles.longitude_max = np.pi

# Define the camera rotation.
# The rotation follows the rule of the right hand.
bpy.context.object.rotation_euler[0] = poses[0]*np.pi/180            # Along x.
bpy.context.object.rotation_euler[1] = poses[1]*np.pi/180            # Along y.
bpy.context.object.rotation_euler[2] = poses[2]*np.pi/180            # Along z.

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# Assigne the new camera to the scene.
bpy.context.scene.camera = cam

# Activate the use of nodes.
bpy.context.scene.use_nodes = True

# Render an image for any pair ((width, height), (x, y, z)).
for width, height in resolutions:
    for x, y, z in camera_coordinates:
        
        # Set the camera parameters.
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.camera.location.x = x
        bpy.context.scene.camera.location.y = y
        bpy.context.scene.camera.location.z = z
        tree = bpy.context.scene.node_tree
        links = tree.links
        rl = tree.nodes.new(type="CompositorNodeRLayers")
        
        # Depth map.
        fileDepthOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        fileDepthOutput.format.file_format = 'OPEN_EXR'
        fileDepthOutput.base_path = 'outputImages/{w}_{h}'.format(w=width, h=height)
        fileDepthId = 'test_{x}_{y}_{z}_{w}_{h}_depth_'.format(x=x, y=y, z=z, w=width, h=height)
        fileDepthPath = '{}/{}.exr'.format(fileDepthOutput.base_path, fileDepthId)
        fileDepthOutputSocket = fileDepthOutput.file_slots.new(fileDepthId)
        links.new(rl.outputs['Depth'], fileDepthOutputSocket)
        
        # Texture.
        fileTextureOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        fileTextureOutput.format.file_format = 'PNG'
        fileTextureOutput.base_path = 'outputImages/{w}_{h}'.format(w=width, h=height)
        fileTextureOutputId = 'test_{x}_{y}_{z}_{w}_{h}_'.format(x=x, y=y, z=z, w=width, h=height)
        fileTextureOutputPath = '{}/{}.png'.format(fileTextureOutput.base_path, fileTextureOutputId)
        fileTextureOutputSocket = fileTextureOutput.file_slots.new(fileTextureOutputId)
        links.new(rl.outputs['Image'], fileTextureOutputSocket)
        
        # Launch the rendering.
        bpy.ops.render.render(write_still=False)
        
        # Clean the created nodes.
        bpy.context.scene.node_tree.nodes.remove(fileTextureOutput)
        bpy.context.scene.node_tree.nodes.remove(fileDepthOutput)

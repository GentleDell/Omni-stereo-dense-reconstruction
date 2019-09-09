Given a blender file "file_name.blender", this script permits to place a 360 degree cameras with arbitrary pose in the scene and get the picture. This render script is only tested on blender2.79.

To launch the rendering: <br>
    1. Edit the render.cfg file (next to this scrip). If there is no render.cfg then create one and put it in the folder containing this script. In render.cfg, You can set the pose of the 360 camera, resolution of output image and in which GPU you want to render the image. An example is provided below:
'''    
    # There are 4 gpus in total and we want to use gpu0 to render the 
    # scene. The camera is placed at (X=10, Y=30, Z=1) and rotated by 
    # 90 degree around the X-axis. Resolution is (1024,512)
    all_gpu 4
    gpu 0
    poses 90 0 0 10 30 1
    res 1024 512
''' 
    2. If blender has been downloaded to or installed in your system, then open a terminal, enter the folder containing this script ("render.py"), and run the following command:
            
        /path/to/blender_folder/blender -b scene_name.blender -P render.py
        
The images will be saved in the folder "outputImages", within the same folder of this script.

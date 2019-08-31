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

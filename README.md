# Multi-view 360 Stereo

## About
This project is an python implemetation of multiview 360 stereo. 

To deal with distortions of 360 images and to estimate depth maps, cubic projection is implemented. It decomposes every 360 images into 6 views which can be treated as regular images.Then [patchmatching stereo](https://www.microsoft.com/en-us/research/publication/patchmatch-stereo-stereo-matching-with-slanted-support-windows/) implemented in the [COLMAP](https://colmap.github.io/) is used to estimate depth. To improvethe performance of depth estimation and leverage the rich textures captured by 360 cameras,a view selection method based on similarity and triangulation angle is implemented. Finally,to generate new views at arbitrary poses, a view synthesis algorithm is accomplished, whereindices volumes and costs volumes are defined for pixel-wise selection and texture synthesis. Details of the implementation can be found in (this document)[].

## How to Use
### Prerequisites

- [Ubuntu](http://releases.ubuntu.com/18.04/) or other Linux system, Mac. <br>
  As the installtions of some packages do not work well on Windows, this project is only tested on Ubuntu 18.04.
  

- To use cubemap functions, you have to install: <br>
  [numpy](https://www.numpy.org/), 
  [matplotlib](https://matplotlib.org/), 
  [scipy](https://www.scipy.org/), 
  [OpenCV](https://opencv.org), 
  [pyshtools](https://pypi.org/project/pyshtools/4.0/),
  [interpolation](https://pypi.org/project/interpolation/).
  

- To execute the wrapper correctly, a computer equiped with Nvidia GPUs (at least 4GB memory) with [CUDA](https://developer.nvidia.com/cuda-downloads) installed is necessary. It is recommended to execute the reconstruction on a server to avoid the 'timeout' error from GPU. Besides, you have to install [colmap](https://colmap.github.io/) or use the colmap in [this repository](https://github.com/GentleDell/Omni-stereo-dense-reconstruction/tree/master/PatchMatchStereo_GPU).

### Examples

An examples can be found [here](), includeing :
1. a simple introduction of the provided cubemap class;
2. an example for the wrapper of 360 depth map estimation;
3. an example for the views selection;
4. an example for the view synthesis. 


## File Formats

Each workspaec folder is orgnized as follow:
```
workspace                                   # can be set by "--workspace=/path/to/workspace"
│
├── cubemaps				    # contains all cubic maps, [backward, left, forward, right, up and down], as well as their parameters.
│   │
│   ├── parameters 			    	# contains parameters
│   │   │
│   │   ├── view0					# paramenters for the view0 -- backward view
│   │   │   ├── cameras.txt					# camera model
│   │   │   ├── image.txt					# views' poses 
│   │   │   └── points3D.txt                                    # empty file
│   │   ├── view1					# paramenters for the view1 -- left view
│   │   │   ├── cameras.txt
│   │   │   ├── image.txt
│   │   │   └── points3D.txt
│   │   ├── ...
│   │   │ 
│   │   └── view5					# paramenters for the view5 -- downward view
│   │       └── ...
│   │ 
│   ├── view0					# all backward views
│   │   ├── cam360\_1_view0.png				# backward view from camera 1
│   │   ├── cam360\_2_view0.png				# backward view from camera 2
│   │   └── ...
│   ├── view1
│   │	├── cam360\_1_view1.png                  # all left views
│   │   ├── cam360\_2_view1.png
│   │   └── ...
│   ├── ...
│   │ 
│   └── view5 					# all downward views
│       └── ...
│
├── omni_depthmap			    # estimated 360 depth maps and cost maps
│   │
│   ├── cost_maps				# cost maps
│   │   │
│   │   ├── cam360\_1                        		# maps for camera360_1
│   │   │   ├── 360\_cost\_maps.exr				# cost maps for camera360_1
│   │   │   ├── cam360\_1\_view0.geometric.bin			# cost maps for the view0 of camera360_1
│   │   │   ├── cam360\_1\_view1.geometric.bin			# cost maps for the view1 of camera360_1
│   │   │   ├── ...
│   │   │   └── cam360\_1\_view5.geometric.bin			# cost maps for the view5 of camera360_1
│   │   │
│   │   ├── cam360\_2					# maps for camera360_2
│   │   │   ├── 360\_cost\_maps.exr				# cost maps for camera360_2
│   │   │   ├── cam360\_2\_view0.geometric.bin			# cost maps for the view0 of camera360_2
│   │   │   ├── cam360\_2\_view1.geometric.bin			# cost maps for the view1 of camera360_2
│   │   │   ├── ...
│   │   │   └── cam360\_2\_view5.geometric.bin			# cost maps for the view5 of camera360_2
│   │   └── ...
│   │ 
│   └── depth_maps				# depth maps, with the same structure as the above cost_maps
│       │
│       ├── cam360_1
│       │   ├── 360_cost_maps.exr
│       │   ├── cam360\_1\_view0.geometric.bin
│       │   ├── cam360\_1\_view1.geometric.bin
│       │   ├── ...
│       │   └── cam360\_1\_view5.geometric.bin
│       ├── cam360_2
│       │   ├── 360\_cost_maps.exr
│       │   ├── cam360\_2\_view0.geometric.bin
│       │   ├── cam360\_2\_view1.geometric.bin
│       │   ├── ...
│       │   └── cam360\_2\_view5.geometric.bin
│       └── ...
│
├── patch_match_ws			    # workspace for the patchmatchingstereo_GPU in colmap
│   │
│   └── colmap workspace
│
└── cam360.pickle                           # save all depth and cost for debug and view synthesis
```

### Image Files
All views are stored in the `cubemaps` folder and grouped by their orientaions. We name each view according to it's source 360 image and their view index. For example, the forward view of theh 3rd 360 camera will be named 'cam360\_3\_view2.png'.

### Camera Poses
In cam360 class, rotation R is from WORLD to CAMERA; translation t is from CAMERA to WORLD and is expressed under camera coordinates. Therefore, [R, t] describes the movement from the world coordinte to the camera coordinate. In other words, given a point Pw from world coordinate, the corresponding coordinate P\_cam is P\_cam = [R | t] * Pw 

### Output Format
The dense_reconstruction.py generate the omni\_depthmap folder and save all cost maps and depth maps to it. Both of depth and cost maps are stored in .exr file which can be read by opencv imread, setting cv2.IMREAD\_ANYDEPTH. 

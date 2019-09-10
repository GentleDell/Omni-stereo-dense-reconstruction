# Multi-view 360 Stereo

## About
This project is a python implemetation of multiview 360 stereo. Details can be found in [this report](https://github.com/GentleDell/Omni-stereo-dense-reconstruction/blob/master/Multiview%20360%20Stereo.pdf).

To deal with distortions of 360 images and to estimate depth maps, cubic projection is implemented. It decomposes every 360 images into 6 views which can be treated as regular images. Then [PatchMatching Stereo](https://www.microsoft.com/en-us/research/publication/patchmatch-stereo-stereo-matching-with-slanted-support-windows/) implemented in the [COLMAP](https://colmap.github.io/) is used to estimate depth. To improve the performance of depth estimation and leverage the rich textures captured by 360 cameras,a view selection method based on similarity and triangulation angle is implemented. Finally, to generate new views at arbitrary poses, a view synthesis algorithm is accomplished, where indices volumes and costs volumes are defined for pixel-wise selection and texture synthesis.

## How to Use
### Prerequisites

- [Ubuntu](http://releases.ubuntu.com/18.04/) or other Linux system, Mac. <br>
  As the installations of some packages do not work well on Windows, this project is only tested on Ubuntu 18.04.
  

- To use this project, you have to install: <br>
  [numpy](https://www.numpy.org/), 
  [matplotlib](https://matplotlib.org/), 
  [scipy](https://www.scipy.org/), 
  [OpenCV](https://opencv.org), 
  [pyshtools](https://pypi.org/project/pyshtools/4.0/),
  [interpolation](https://pypi.org/project/interpolation/).
  

- To run dense reconstruction correctly, a computer with Nvidia GPUs equipped and [CUDA](https://developer.nvidia.com/cuda-downloads) installed is necessary. If high resolution images are used for reconstruction, more GPU memory are required. Besides, for depth estimation, it is enough to install [COLMAP](https://colmap.github.io/). But to execute view synthesis, the modified COLMAP in [this repository](https://github.com/GentleDell/Omni-stereo-dense-reconstruction/tree/master/PatchMatchStereo_GPU) has to be installed.

### Examples

Examples can be found [here](https://github.com/GentleDell/Omni-stereo-dense-reconstruction/blob/master/cube_script/Tutorials.ipynb), including :
1. a simple introduction of the provided cubemap class;
2. an example of 360 depth map estimation;
3. an example of the views selection;
4. an example of the view synthesis. 


## File Formats

Each workspaec folder is orgnized as follow:
```
workspace                   # can be set by "--workspace=/path/to/workspace"
│
├── cubemaps                    # contains all cubic maps, [backward, left, forward, right, up and down], as well as their parameters.
│   │
│   ├── parameters                  # contains parameters
│   │   │
│   │   ├── view0                       # paramenters for the view0 -- backward view
│   │   │   ├── cameras.txt                 # camera model
│   │   │   ├── image.txt                   # views' poses 
│   │   │   └── points3D.txt                # empty file
│   │   ├── view1                       # paramenters for the view1 -- left view
│   │   │   ├── cameras.txt
│   │   │   ├── image.txt
│   │   │   └── points3D.txt
│   │   ├── ...
│   │   │ 
│   │   └── view5                       # paramenters for the view5 -- downward view
│   │       └── ...
│   │ 
│   ├── view0                       # all backward views
│   │   ├── cam360_1_view0.png          # backward view from camera 1
│   │   ├── cam360_2_view0.png          # backward view from camera 2
│   │   └── ...
│   ├── view1                       # all left views
│   │	├── cam360_1_view1.png               
│   │   ├── cam360_2_view1.png
│   │   └── ...
│   ├── ...
│   │ 
│   └── view5                       # all downward views
│       └── ...
│
├── omni_depthmap               # estimated 360 depth maps and cost maps
│   │
│   ├── cost_maps                   # cost maps
│   │   │
│   │   ├── cam360_1                    # estimations of camera360_1
│   │   │   ├── 360_cost_maps.exr               # cost maps of camera360_1
│   │   │   ├── cam360_1_view0.geometric.bin    # cost maps of the view0 of camera360_1
│   │   │   ├── cam360_1_view1.geometric.bin    # cost maps of the view1 of camera360_1
│   │   │   ├── ...
│   │   │   └── cam360_1_view5.geometric.bin    # cost maps of the view5 of camera360_1
│   │   │
│   │   ├── cam360_2                    # estimations of camera360_2
│   │   │   ├── 360_cost_maps.exr               # cost maps of camera360_2
│   │   │   ├── cam360_2_view0.geometric.bin    # cost maps of the view0 of camera360_2
│   │   │   ├── cam360_2_view1.geometric.bin    # cost maps of the view1 of camera360_2
│   │   │   ├── ...
│   │   │   └── cam360_2_view5.geometric.bin    # cost maps of the view5 of camera360_2
│   │   └── ...
│   │ 
│   └── depth_maps              # depth maps, with the same structure as the above cost_maps folder
│       │
│       ├── cam360_1
│       │   ├── 360_depth_maps.exr
│       │   ├── cam360_1_view0.geometric.bin
│       │   ├── cam360_1_view1.geometric.bin
│       │   ├── ...
│       │   └── cam360_1_view5.geometric.bin
│       ├── cam360_2
│       │   ├── 360_depth_maps.exr
│       │   ├── cam360_2_view0.geometric.bin
│       │   ├── cam360_2_view1.geometric.bin
│       │   ├── ...
│       │   └── cam360_2_view5.geometric.bin
│       └── ...
│
├── patch_match_ws              # workspace for the patchmatchingstereo_GPU in colmap
│   │
│   └── colmap workspace
│
└── cam360.pickle               # save all depth estimtations and cost maps for debugging and view synthesis
```

### Image Files
All views are stored in the `cubemaps` folder and grouped by their orientations. We name each view according to the index of its source 360 image and their view index. For example, the forward view (view 2) of the 3rd 360 camera will be named 'cam360_3_view2.png'.

### Camera Poses
In cam360 class, rotation R is from WORLD to CAMERA; translation t is from CAMERA to WORLD and is expressed under camera coordinates. Therefore, [R, t] describes the movement from the world coordinate to the camera coordinate. In other words, given a point Pw from world coordinate, the corresponding coordinate P_cam is P_cam = [R | t] * Pw 

### Output Format
The dense_reconstruction.py generate the omni_depthmap folder and save all cost maps and depth maps to it. Both of depth and cost maps are stored in .exr file which can be read by [OpenCV](https://opencv.org) imread, with cv2.IMREAD_ANYDEPTH being set. 

## Possible Errors
**1.'timeout' error from GPU when running this project on command line.**<br>
The reason can be found [here](https://colmap.github.io/faq.html#fix-gpu-freezes-and-timeouts-during-dense-reconstruction). You can run this project in Jupyter notebook, provided GPU memory is enough. If GPU memory is not enough, it is recommended to run this project on a server.

**2. Depth map has only one/two/three etc. views.**<br>
Firstly, please check the variable "views_for_depth". It should be set to 4 or 6. 

Then please make sure the input poses are correct i.e. from world to local when using the command line; from camera to world when using "dense_from_cam360list()". 

Finally, if you enabled view selection, please try to disable view selection and run it again, since the view selection needs rich texture to run correctly.

**3.It reports an error after printing "Reprojecting cost maps ..."**<br>
Please check whether the original colmap is used for dense reconstruction. Since the original colmap does not output cost maps, there will be errors when projecting cost maps to 360camera sphere. But this error does not affect the depth.

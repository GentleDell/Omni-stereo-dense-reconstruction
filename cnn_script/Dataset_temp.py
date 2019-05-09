# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:13:41 2019

@author: Gentle Deng
"""

import numpy as np

class Stereo_Dataset:
    def __init__(self, path: str = None, read_only: bool=True, patch_size: list = [9*9]):
        
        if path is None:
            raise ValueError("Input ERROR! Invalid path!")
            
        """ Stereo Dataset. """
  
    def __init__(self, data_path, receptive_size, max_disp, transform=None):
        """
        Args:
            data_path (string): Path to KITTI dataset
            transform (callable, optional): Optional transform to be applied.
        """
        self.data = glob.glob(os.path.join(data_path, 'disp_noc', '*_10.png'))
        self.hw = (receptive_size - 1) // 2  # half-width
        self.max_disp = max_disp
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ###############
        ## DISPARITY ##
        ###############
        # read disparity map
        path = self.data[idx]
        disp_map = cv2.imread(path, 0)
        # list of non-zero indicies
        disp_locs = np.nonzero(disp_map)
        # random non-zero index
        r = random.randint(0, len(disp_locs[0])-1)
        p_y, p_x = disp_locs[0][r], disp_locs[1][r]
        # get disparity value
        d = disp_map[p_y, p_x]
        # clip at `max_disp'
        d = min(d, self.max_disp)

        ##########
        ## LEFT ##
        ##########
        # read left image
        path = path.replace('disp_noc', 'image_1')
        left_img = cv2.imread(path, 0)
        # create left patch (padded if necessary)
        left_padded = np.zeros((1, self.hw*2+1, self.hw*2+1), dtype=np.float32)
        # extract left patch (subtrack disparity from `p_x' for left coords)
        left_patch = left_img[p_y-self.hw: p_y+self.hw+1, 
                p_x-d-self.hw: p_x-d+self.hw+1]
        # normalize [0-1]
        left_patch = left_patch / 255
        # store in padded array
        left_padded[:, :left_patch.shape[0], :left_patch.shape[1]] = left_patch

        ###########
        ## RIGHT ##
        ###########
        # read right image
        path = path.replace('image_1', 'image_0')
        right_img = cv2.imread(path, 0)
        # create right patch (padded if necessary)
        right_padded = np.zeros((1, self.hw*2+1, self.max_disp+self.hw*2+1),
                dtype=np.float32)
        # extract right patch
        right_patch = right_img[p_y-self.hw: p_y+self.hw+1,
                p_x-self.hw: p_x+self.hw+self.max_disp+1]
        # normalize [0-1]
        right_patch = right_patch / 255
        # store in padded array
        right_padded[:, :right_patch.shape[0], 
                :right_patch.shape[1]] = right_patch

        return left_padded, right_padded, int(d)
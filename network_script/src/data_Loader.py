from __future__ import print_function
import os
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import datasets, transforms

class dataLoader(object):
    def __init__(self, data_directory, train_val_split_dir, dataset_name, psz,
                 half_range, image_height, image_width, mode='Train'):
        self.psz  = psz
        self.mode = mode
        self.half_range    = half_range
        self.image_height  = image_height
        self.image_width   = image_width
        self.dataset_name  = dataset_name
        self.data_directory = data_directory
        self.train_val_split_dir = train_val_split_dir
        self.get_data()

    def get_data(self):
        points       = []
        rgb_images_l = []
        rgb_images_r = []

        # Full images file path
        file_path = os.path.join(self.train_val_split_dir, self.dataset_name)

        with open(file_path, 'r') as f:
            points_all = f.readlines()

        for point in points_all:
            if point == '\n':
                continue

            tr_num, _, l_center_x, l_center_y, r_center_x = point.split(',')
            # Removes '\n' at the begining
            r_center_x = r_center_x[:-1]

            tr_num     = int(tr_num)
            l_center_x = int(l_center_x)
            l_center_y = int(l_center_y)
            r_center_x = int(r_center_x)
            r_center_y = int(l_center_y)

            points.append((tr_num, l_center_x, l_center_y, r_center_x, r_center_y))

        l_images_all = glob.glob(os.path.join(self.data_directory, 'image_2/*_10.png'))
        r_images_all = glob.glob(os.path.join(self.data_directory, 'image_3/*_10.png'))
        
        try:
            assert len(l_images_all) == len(r_images_all)
        except AssertionError:
            print('Un-equal left and right stereo image pairs!')
            raise

        for img in range(len(l_images_all)):
            images_l_path = os.path.join(self.data_directory,\
                                          'image_2/%06d_10.png' % (img))
            images_r_path = os.path.join(self.data_directory,\
                                          'image_3/%06d_10.png' % (img))
            
            # read images into tensor
            image_l = Image.open(images_l_path)
            # modified by zhantao deng  
            # to make sure the sizes of images satisfy the requirement 
            W, H = image_l.size
            if H != self.image_height or W != self.image_width:
                image_l = image_l.resize((self.image_width, self.image_height))     
            image_l = 255*transforms.ToTensor()(image_l)
            
            image_r = Image.open(images_r_path)
            # modified by zhantao deng  
            # to make sure the sizes of images satisfy the requirement 
            W, H = image_r.size
            if H != self.image_height or W != self.image_width:
                image_r = image_r.resize((self.image_width, self.image_height))     
            image_r = 255*transforms.ToTensor()(image_r)
                
            #debug
            if image_l.cpu().numpy().shape != (3, 375, 1242) or image_r.cpu().numpy().shape != (3, 375, 1242):
                print('the left image size is:', image_l.cpu().numpy().shape)
                print('the left image path is:',images_l_path)
                print('the right image size is:', image_r.cpu().numpy().shape)
                print('the right image path is:',images_r_path)

            rgb_images_l.append(image_l)
            rgb_images_r.append(image_r)

        self.points = points

        if self.mode == 'Test':
            self.max_steps = len(points)

        # Preprocess Image
        all_rgb_images_l, all_rgb_images_r = self.preprocess_image(rgb_images_l, rgb_images_r)

        # Set to std deviation of 1
        self.all_rgb_images_l = all_rgb_images_l
        self.all_rgb_images_r = all_rgb_images_r

        # Free up memory
        del rgb_images_l
        del rgb_images_r

    def preprocess_image(self, rgb_images_l, rgb_images_r):
        all_rgb_images_l = []
        all_rgb_images_r = []

        print('Preprocessing......')
        for i in tqdm(range(len(rgb_images_l))):
            image_l = rgb_images_l[i]
            image_r = rgb_images_r[i]

            # Reduce mean and std
            image_l = (image_l - image_l.mean()) / image_l.std()
            image_r = (image_r - image_r.mean()) / image_r.std()

            all_rgb_images_r.append(image_r)
            all_rgb_images_l.append(image_l)

        print('Preprocess Complete!')

        return all_rgb_images_l, all_rgb_images_r

    def gen_random_data(self):
        while True:
            indices = list(range(len(self.points)))
            random.shuffle(indices)
            for i in indices:
                tr_num, l_center_x, l_center_y, r_center_x, r_center_y = self.points[i]

                yield (tr_num, l_center_x, l_center_y, r_center_x, r_center_y)

    def gen_val_data(self):
        while True:
            indices = range(len(self.points))
            for i in indices:
                tr_num, l_center_x, l_center_y, r_center_x, r_center_y = self.points[i]

                yield (tr_num, l_center_x, l_center_y, r_center_x, r_center_y)


    def gen_data_batch(self, batch_size):
        # Generate data based on training/validation
        if self.mode == 'Train':
            # Randomize data
            data_gen = self.gen_random_data()
        else:
            data_gen = self.gen_val_data()

        while True:
            image_l_batch  = []
            image_r_batch = []

            # Generate training batch
                # ATTENTION: the pytorch will not report errors when the index is out of the bound
            for ct in range(batch_size):               
                
                tr_num, l_center_x, l_center_y, r_center_x, r_center_y = next(data_gen)

                l_image = self.all_rgb_images_l[tr_num]
                r_image = self.all_rgb_images_r[tr_num]

                # Get patches
                l_patch = l_image[:, l_center_y-self.psz:l_center_y+self.psz+1,\
                                     l_center_x-self.psz:l_center_x+self.psz+1]

                r_patch = r_image[:, r_center_y-self.psz:r_center_y+self.psz+1,\
                                     r_center_x-self.half_range-self.psz:r_center_x+self.half_range+self.psz+1]   
                
                # debug
                # some weired dimension error when executed on the server, here we check and skip the errors              
                if r_patch.cpu().numpy().shape != (3,37,237) or l_patch.cpu().numpy().shape != (3,37,37):    
                    print("Bad patch size:",r_patch.shape , l_patch.shape)
                    
                    
                
                # Append to generated batch
# This original implementation is not correct as pytorch can not convert a list of tensor to be a tensor
#                l_patch = l_patch.unsqueeze(0)
#                r_patch = r_patch.unsqueeze(0)
#            image_l_batch = torch.tensor(image_l_batch)
#            image_r_batch = torch.tensor(image_r_batch)
#           
                #  modified by zhantao deng
                image_l_batch.append(l_patch)
                image_r_batch.append(r_patch)

            image_l_batch = torch.stack(image_l_batch)
            image_r_batch = torch.stack(image_r_batch)
                    
                    
            yield image_l_batch, image_r_batch

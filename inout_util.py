# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""

import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import dicom

class DataLoader(object):
    def __init__(self, args, sample_ck=False):
        self.phase = args.phase
        self.extension = args.extension
        #data directory
        input_path_list = np.array(sorted(glob(os.path.join(\
                args.dcm_path, '*', args.input_path, '*.' + args.extension), recursive=True)))
        target_path_list = np.array(sorted(glob(os.path.join(\
                args.dcm_path, '*', args.target_path, '*.' + args.extension), recursive=True)))
        
        np.random.seed(args.seed)
        shffle_inx = list(range(len(input_path_list)))
        np.random.shuffle(shffle_inx)
        n_train = int(len(shffle_inx)*args.train_ratio)
        
        train_idx, test_idx = shffle_inx[:n_train], shffle_inx[n_train:]
        if args.phase == 'train':
            self.input_path_list = list(input_path_list[train_idx])
            self.target_path_list = list(target_path_list[train_idx])
        elif args.phase == 'test':
            self.input_path_list = list(input_path_list[test_idx])
            self.target_path_list = list(target_path_list[test_idx])

        self.data_index = list(range(len(self.input_path_list)))
        assert len(self.input_path_list) == len(self.target_path_list), \
                'the number of samples of input data & target data should be same'
        
        #save directory        
        self.model_save_dir = os.path.join(args.result, args.checkpoint_dir)
        self.tfboard_save_dir = os.path.join(args.result, args.log_dir)
        self.inf_save_dir = os.path.join(args.result, args.inference_result)
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.tfboard_save_dir):
            os.makedirs(self.tfboard_save_dir)
        if not os.path.exists(self.inf_save_dir):
            os.makedirs(self.inf_save_dir)

        print('save directories\n  model checkpoint : {}\n  inference result : {}\n  tensor board : {}'.\
      format(self.model_save_dir, self.inf_save_dir, self.tfboard_save_dir))
        
        #image params
        self.depth = args.depth
        self.image_size = args.patch_size
        self.whole_size = args.whole_size
        self.trun_max = args.trun_max
        self.trun_min = args.trun_min
        self.patch_per_img = args.patch_per_img
        
        #loader params
        self.prefetch_buffer = args.prefetch_buffer
        self.num_parallel_calls = args.num_parallel_calls
        self.batch_size = args.batch_size
        self.norm = args.norm
        self.augument = args.augument
        self.is_unpair = args.is_unpair 
        self.count_get_images_fun = 0
        self.sample_ck= sample_ck
        if str(args.norm).lower() == 'n-11':
            self.psnr_range = 2 
        elif str(args.norm).lower() == 'n01':
            self.psnr_range = 1
        else:self.psnr_range =args.truc_max - args.truc_min
        
        if (sample_ck) or (args.phase=='test'):
            self.input_path_list = list(input_path_list[test_idx])
            self.target_path_list = list(target_path_list[test_idx])
            self.data_index = list(range(len(self.input_path_list)))
            
            self.batch_size = 1
            self.image_size = args.whole_size
            self.augument = False
            self.is_unpair = False
            self.prefetch_buffer = 4
            self.num_parallel_calls = 4
        
    def loader(self):
        dataset = tf.data.Dataset.from_generator(self.data_generaotr, (tf.string, tf.string))
        dataset = dataset.map(lambda input_, target_: tuple(tf.py_func(
                    self.get_images, [input_, target_], [tf.float32, tf.float32])),\
                    num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.prefetch(self.prefetch_buffer).repeat()
        dataset = dataset.map(self.set_images_shape)
        dataset = dataset.batch(self.batch_size)
        iter_ = dataset.make_one_shot_iterator()
        element = iter_.get_next()
        return element

    def data_generaotr(self):
        if self.phase == 'train':
            while True:
                input_idx = np.random.choice(self.data_index)
                target_idx = np.random.choice(self.data_index) \
                                if self.is_unpair else input_idx
                yield (self.input_path_list[input_idx], self.target_path_list[target_idx])
        else:
            for i in self.data_index:
                yield (self.input_path_list[i], self.target_path_list[i])
        
    def get_images(self, input_dir, target_dir):
        def get_pixels_hu(slice_):
            image = slice_.pixel_array
            image = image.astype(np.int16)
            image[image == -2000] = 0
            intercept = slice_.RescaleIntercept
            slope = slice_.RescaleSlope
            if slope != 1:
                image = slope * image.astype(np.float32)
                image = image.astype(np.int16)
            image += np.int16(intercept)
            
            return np.array(image, dtype=np.int16)
        
        in_, tar_ = input_dir.decode(), target_dir.decode()
        if ((self.count_get_images_fun % self.patch_per_img == 0) and (self.phase=='train')) or (self.phase == 'test') or (self.sample_ck):
            self.count_get_images_fun = 0
            if self.extension in ['IMA', 'DCM']:
                self.input_org = get_pixels_hu(dicom.read_file(in_))
                self.target_org = get_pixels_hu(dicom.read_file(tar_))
            elif self.extension == 'npy':
                self.input_org = np.load(in_)
                self.target_org =  np.load(tar_)

        input_ = self.normalize(self.input_org, self.trun_max , self.trun_min)
        target_  = self.normalize(self.target_org, self.trun_max , self.trun_min)
        
        if self.image_size != self.whole_size:
            input_, target_ = self.get_randam_patches(input_, target_)
        return np.expand_dims(input_, axis=-1), np.expand_dims(target_, axis=-1)
    
    def set_images_shape(self, input_, target_ ):
        input_.set_shape([None, None, self.depth])
        target_.set_shape([None, None, self.depth])
    
        # resize to model input size
        input_resized = tf.image.resize_images(input_, [self.image_size, self.image_size])
        target_resized  = tf.image.resize_images(target_, [self.image_size, self.image_size])
        return input_resized, target_resized  
    
    
    def normalize(self, img, max_=3072, min_=-1024):
        img = img.astype(np.float32) 
        img[img > max_] = max_
        img[img < min_] = min_
        if str(self.norm).lower()  == 'n-11':  #-1 ~ 1
            img = 2 * ((img - min_) / (max_  -  min_)) -1
            return img
        elif str(self.norm).lower()  == 'n01':  #0 ~ 1
            img = (img - min_) / (max_  -  min_)
            return img
        else:
            return img

    def get_randam_patches(self, LDCT_slice, NDCT_slice):
        def augumentation(LDCT, NDCT):  #REDCNN
            sltd_random_indx=  [np.random.choice(range(4)), np.random.choice(range(2))]
            if sltd_random_indx[0] ==0 : 
                return rotate(LDCT, 45, reshape = False), rotate(NDCT, 45, reshape = False)
            elif sltd_random_indx[0] ==1 :
                param  = [True, False][sltd_random_indx[1]]
                if param:
                    return LDCT[:, ::-1], NDCT[:, ::-1] #horizontal
                return LDCT[::-1, :], NDCT[::-1, :] # vertical 
            elif sltd_random_indx[0] ==2 :
                param  = [0.5, 2][sltd_random_indx[1]]
                return LDCT * param, NDCT * param
            elif sltd_random_indx[0] ==3 :
                return LDCT, NDCT
            
            
        whole_h =  whole_w = self.whole_size
        h = w = self.image_size

        #patch image range
        hd, hu = h//2, int(whole_h - np.round(h/2))
        wd, wu = w//2, int(whole_w - np.round(w/2))

        #patch image center(coordinate on whole image)
        h_pc, w_pc  = np.random.choice(range(hd, hu+1)), np.random.choice(range(wd, wu+1))
        if len(LDCT_slice.shape) == 3: # 3d patch
            LDCT_patch = LDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)),\
                                    w_pc - wd: int(w_pc + np.round(h / 2))]
            NDCT_patch = NDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)), \
                                    w_pc - wd: int(w_pc + np.round(h / 2))]
        else: # 2d patch
            LDCT_patch = LDCT_slice[h_pc - hd : int(h_pc + np.round(h/2)), \
                                    w_pc - wd : int(w_pc + np.round(h/2))]
            NDCT_patch = NDCT_slice[h_pc - hd : int(h_pc + np.round(h/2)), \
                                    w_pc - wd : int(w_pc + np.round(h/2))]

        if self.augument:
            return augumentation(LDCT_patch, NDCT_patch)
        return LDCT_patch, NDCT_patch


#psnr
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))


#---------------------------------------------------
# argparser string -> boolean type
def ParseBoolean (b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError ('Cannot parse string into boolean.')
        
# argparser string -> boolean type
def ParseList(l):
    return l.split(',')
       

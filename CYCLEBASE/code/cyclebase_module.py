# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""
import tensorflow as tf
import numpy as np
import copy
import math
"""
def Loss
"""
def mean_square(A, B):
    return tf.reduce_mean((A - B)**2)

def mean_abs(A, B):
    return tf.reduce_mean(tf.abs(A - B))
    
def adv_gen_loss(D_GA, D_FB):
    Gen_loss_A2B = mean_square(D_GA, tf.ones_like(D_GA)) 
    Gen_loss_B2A = mean_square(D_FB, tf.ones_like(D_FB)) 
    adversarial_loss = Gen_loss_A2B + Gen_loss_B2A
    return adversarial_loss, Gen_loss_A2B, Gen_loss_B2A

def cycle_loss(A, F_GA, B, G_FB, lambda_):
    return lambda_ * (mean_abs(A, F_GA) + mean_abs(B, G_FB))

def identity_loss(A, F_A, B, G_B, gamma_):
    return   gamma_ * (mean_abs(F_A, A) + mean_abs(G_B, B))

def adv_disc_loss(inputs):
    D_A, D_sample_FB, D_B, D_sample_GA = inputs
    D_loss_real_A = mean_square(D_A, tf.ones_like(D_A))
    D_loss_Gen_FB = mean_square(D_sample_FB, tf.zeros_like(D_sample_FB))
    D_loss_real_B = mean_square(D_B, tf.ones_like(D_B))
    D_loss_Gen_GA = mean_square(D_sample_GA, tf.zeros_like(D_sample_GA))
    D_loss_A = (D_loss_real_A + D_loss_Gen_FB)/2
    D_loss_B = (D_loss_real_B + D_loss_Gen_GA)/2
    D_loss =  D_loss_A + D_loss_B
    
    return D_loss, D_loss_real_A, D_loss_Gen_FB, \
            D_loss_real_B, D_loss_Gen_GA, D_loss_A, D_loss_B

"""
def Summary
"""
def genrator_loss_summary(inputs):
    Gen_loss, Gen_loss_A2B, Gen_loss_B2A,cycle_loss, identity_loss, \
    lambda_, gamma_ = inputs
    
    Gen_loss_sum = tf.summary.scalar("1_G_loss", Gen_loss, \
        family = 'Generator_loss')
    Gen_loss_A2B_sum = tf.summary.scalar("2_adv_loss_A2B", Gen_loss_A2B, \
        family = 'Generator_loss')
    Gen_loss_B2A_sum = tf.summary.scalar("3_adv_loss_B2A", Gen_loss_B2A, \
        family = 'Generator_loss')
    generator_loss_list = [Gen_loss_sum, Gen_loss_A2B_sum, Gen_loss_B2A_sum]
    
    if lambda_ > 0:
        cycle_loss_sum = tf.summary.scalar("5_cycle_loss", cycle_loss, \
        family = 'Generator_loss')
        generator_loss_list += [cycle_loss_sum]
    if gamma_ > 0:
        identity_loss_sum = tf.summary.scalar("6_identity_loss", identity_loss, \
        family = 'Generator_loss')
        generator_loss_list += [identity_loss_sum]
    return tf.summary.merge(generator_loss_list)

    
def discriminator_loss_summary(inputs):    
    D_loss, D_loss_A, D_loss_real_A, D_loss_Gen_FB, \
    D_loss_B, D_loss_real_B, D_loss_Gen_GA = inputs
    D_loss_sum = tf.summary.scalar("0_D_loss", D_loss, \
                family = 'Discriminator_loss')
    D_loss_A_sum = tf.summary.scalar("1_D_loss_A", D_loss_A, \
                family = 'Discriminator_loss')
    D_loss_real_A_sum = tf.summary.scalar("2_D_loss_real_A", D_loss_real_A, \
                family = 'Discriminator_loss')
    D_loss_Gen_FB_sum = tf.summary.scalar("3_D_loss_fake_A(FB)", D_loss_Gen_FB, \
                family = 'Discriminator_loss')
    D_loss_B_sum = tf.summary.scalar("4_D_loss_B", D_loss_B, \
                family = 'Discriminator_loss')
    D_loss_real_B_sum = tf.summary.scalar("5_D_loss_real_B", D_loss_real_B, \
                family = 'Discriminator_loss')
    D_loss_Gen_GA_sum = tf.summary.scalar("6_D_loss_fake_B(GA)", D_loss_Gen_GA, \
                family = 'Discriminator_loss')
    
    summary_list = [D_loss_sum, D_loss_A_sum, D_loss_real_A_sum, \
          D_loss_Gen_FB_sum, D_loss_B_sum, D_loss_real_B_sum, \
          D_loss_Gen_GA_sum]
    d_sum = tf.summary.merge(summary_list)
    return d_sum


def image_summary(inputs):
    real_A, real_B, G_A, whole_A, whole_B, test_G_A = inputs
    train_img_summary = tf.concat([real_A, real_B, G_A], axis = 2)
    summary_image_1 = tf.summary.image('1_train_image', train_img_summary)
    test_img_summary = tf.concat([whole_A, whole_B, test_G_A], axis = 2)
    summary_image_2 = tf.summary.image('2_test_image', test_img_summary)
    return summary_image_1,summary_image_2

def psnr_summary(inputs):
    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    def tf_psnr(img1, img2, PIXEL_MAX = 255.0):
        mse = tf.reduce_mean((img1 - img2) ** 2 )
        if mse == 0:
            return 100
        return 20 * log10(PIXEL_MAX / tf.sqrt(mse))
    whole_A, whole_B, whole_B, test_G_A, psnr_range = inputs
    summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", \
        tf_psnr(whole_A, whole_B, psnr_range ), family = 'PSNR')  
    summary_psnr_result = tf.summary.scalar("2_psnr_output", \
        tf_psnr(whole_B, test_G_A, psnr_range ), family = 'PSNR') 
    return tf.summary.merge([summary_psnr_ldct, summary_psnr_result])



# get psnr
def psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#### cygle gan image pool
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        N_img = len(image)
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            hist_imgs = []
            for i in range(N_img):
                tmp = copy.copy(self.images[idx])[i]
                self.images[idx][i] = image[i]
                hist_imgs.append(tmp)
            return hist_imgs

        else:
            return image
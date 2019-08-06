# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""


"""
################################################################################

1. discriminator
2. generator
3. pre trained VGG net

"""
import os
import tensorflow as tf
import numpy as np

"""
module (discriminator, generator, pretrained vgg)
"""
def discriminator(image, name="discriminator", reuse = True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('d_conv_1'):
            l1 = lrelu(conv2d(image, 64, ks=3, s=1))
        with tf.variable_scope('d_conv_2'):
            l2 = lrelu(conv2d(l1, 64, ks=3, s=2))
        with tf.variable_scope('d_conv_3'):
            l3 = lrelu(conv2d(l2, 128, ks=3, s=1))
        with tf.variable_scope('d_conv_4'):
            l4 = lrelu(conv2d(l3, 128, ks=3, s=2))
        with tf.variable_scope('d_conv_5'):
            l5 = lrelu(conv2d(l4, 256, ks=3, s=1))
        with tf.variable_scope('d_conv_6'):
            l6 = lrelu(conv2d(l5, 256, ks=3, s=2))
        with tf.variable_scope('d_fc_1'):
            fc1 = lrelu(fcn(l6, 1024))
        with tf.variable_scope('d_fc_2'):
            fc2 = fcn(fc1, 1)
        
        return fc2
    
    
def generator(image, name="generator", reuse = True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('g_conv_1'):
            l1 = relu(conv2d(image, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_2'):
            l2 = relu(conv2d(l1, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_3'):
            l3 = relu(conv2d(l2, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_4'):
            l4 = relu(conv2d(l3, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_5'):
            l5 = relu(conv2d(l4, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_6'):
            l6 = relu(conv2d(l5, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_7'):
            l7 = relu(conv2d(l6, 32, ks=3, s=1))
        with tf.variable_scope('g_conv_8'):
            l8 = relu(conv2d(l7, 1, ks=3, s=1))
        
        return l8


class Vgg19:
    def __init__(self, size = 64, vgg_path = '.'):
        self.size = size
        self.VGG_MEAN = [103.939, 116.779, 123.68]

        vgg19_npy_path = os.path.join(vgg_path, "vgg19.npy")
        self.data_dict  = np.load(vgg19_npy_path, allow_pickle=True, encoding='latin1').item()
        print("npy file loaded")


    def extract_feature(self, rgb):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert green.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert blue.get_shape().as_list()[1:] == [self.size, self.size, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        
        assert bgr.get_shape().as_list()[1:] == [self.size, self.size, 3]


        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, 'pool4')
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        return conv5_4


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu_ = relu(bias)
            return relu_

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.nn.relu(x) 

def tanh(x):
    return tf.nn.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def batchnorm(input_, name="batch_norm", training=True):
    x = tf.layers.batch_normalization(input_, axis=3, epsilon=1e-5, \
        momentum=0.1, training=training, \
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    return x

def conv2d(batch_input, out_channels, ks=4, s=2):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, \
            strides=s, padding="valid", \
            kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return x

def fcn(input_, n_weight, activation_fn=None):
    flat_img = tf.contrib.layers.flatten(input_)
    fc = tf.contrib.layers.fully_connected(flat_img, n_weight, activation_fn=activation_fn)
    return fc

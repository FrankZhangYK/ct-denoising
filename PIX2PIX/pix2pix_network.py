# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu

source : 
    https://github.com/phillipi/pix2pix/blob/master/models.lua
    generator : defineG_unet
    https://github.com/xhujoy/CycleGAN-tensorflow
    descriminator : patchGAN
"""

import tensorflow as tf

"""
def Network
"""
def discriminator(args, input_img, output_img, reuse=True, phase='train', name='discriminator'):
    training = phase == 'train'
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        #[batch, height, width, in_channels * 2]
        input_ = tf.concat([input_img, output_img], axis=-1)

        with tf.variable_scope('first_layer'):    
            x = conv2d(input_, args.ndf, ks=4, s=2)
            x = lrelu(x)
        with tf.variable_scope('conv_layer1'):        
            x = conv2d(x, args.ndf*2, ks=4, s=2)
            x = batchnorm(x, training=training)
            x = lrelu(x)
        with tf.variable_scope('conv_layer2'):        
            x = conv2d(x, args.ndf*4, ks=4, s=2)
            x = batchnorm(x, training=training)
            x = lrelu(x)
        with tf.variable_scope('conv_layer3'):        
            x = conv2d(x, args.ndf*8, ks=4, s=1)
            x = batchnorm(x, training=training)
            x = lrelu(x)
        with tf.variable_scope('last_layer'):        
            x = conv2d(x, args.depth, ks=4, s=1)
            x = sigmoid(x)
        return x

def generator(args, image, reuse=True, phase='train', name="generator", N_encoder_layers=8):
    training = phase == 'train'
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
                
        layers = []
        
        # encoder
        i_layer=0
        en_layer_c = [args.ngf, args.ngf*2, args.ngf*4] + [args.ngf*8]*5
        with tf.variable_scope("encoder_"+str(i_layer + 1)):
            x = conv2d(image, en_layer_c[i_layer], padding='same')
            layers.append(x)
            
        i_layer+=1
        with tf.variable_scope("encoder_"+str(i_layer + 1)):  
            x = lrelu(x, 0.2)
            x = conv2d(x, en_layer_c[i_layer], ks=4, s=2, padding='same')
            x = batchnorm(x, training)
            layers.append(x)
            
        i_layer+=1 
        with tf.variable_scope("encoder_"+str(i_layer + 1)):  
            x = lrelu(x, 0.2)
            x = conv2d(x, en_layer_c[i_layer], ks=4, s=2, padding='same')
            x = batchnorm(x, training)
            layers.append(x)
            
        for ei in range(i_layer+1, N_encoder_layers):
            i_layer+=1
            with tf.variable_scope("encoder_"+str(i_layer + 1)):  
                x = lrelu(x, 0.2)
                x = conv2d(x, en_layer_c[i_layer], ks=4, s=2, padding='same')
                if ei < N_encoder_layers-1:
                    x = batchnorm(x, training)
                layers.append(x)
                
        #decoder                 
        N_decoder_layers = N_encoder_layers
        N_layers = N_encoder_layers + N_decoder_layers
        de_layer_c = [args.ngf*8]*4 + [args.ngf*4, args.ngf*2, args.ngf, 1]
        for di in range(N_decoder_layers):
            i_layer +=1
            with tf.variable_scope("decoder_" + str(i_layer - N_encoder_layers + 1)):
                if di != 0:
                    skip_layer = layers[N_layers - i_layer - 1]
                    x = tf.concat([x, skip_layer], axis=3)
                x = relu(x)
                x = deconv2d(x, de_layer_c[di], ks=4, s=2, padding='same')
                if di < N_decoder_layers - 1:
                    x = batchnorm(x, training)
                drop_out = 0.5 if di < 3 else 1
                x= tf.nn.dropout(x, keep_prob=drop_out)

        # output layer
        with tf.variable_scope("ouput_layer"):
            x= tanh(x)
        return x
    


"""
def Network component
"""
def conv2d(batch_input, out_channels, ks=4, s=2, padding='valid'):
    if padding=='valid':
        batch_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(batch_input, out_channels, kernel_size=ks, \
            strides=s, padding=padding, \
            kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return x

                                   
def deconv2d(batch_input, out_channels, ks=4, s =2, padding='valid'):
    x = tf.layers.conv2d_transpose(batch_input, out_channels, \
                                   kernel_size=ks, strides=s, padding=padding, \
                                   kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return x

def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, \
                                         momentum=0.1, training=True, \
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.nn.relu(x) 

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.tanh(x)
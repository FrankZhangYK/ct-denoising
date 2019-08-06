# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""
import tensorflow as tf

"""
def Network
"""
def discriminator(args, image, reuse=False, phase='train', name='discriminator'):
    if args.strct == 'ident':
        training = phase == 'train'

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
      
            with tf.variable_scope('first_layer'):    
                x = conv2d(image, args.ndf, ks=4, s=2)
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
                x = tf.contrib.layers.flatten(x) 
                x = tf.contrib.layers.fully_connected(x, int(x.get_shape()[-1]))
            return x
               

def generator(args, image, reuse=False, phase='train', name="generator"):
    if args.strct == 'ident':
        training = phase == 'train'
        def conv_layer(input_, out_channels, ks=3, s=1, name='conv_layer'):
            with tf.variable_scope(name):
                x = conv2d(input_, out_channels, ks=ks, s=s)
                x = batchnorm(x, training=training)
                x = relu(x)  
                return x
        def gen_module(input_,  out_channels, ks=3, s=1, name='gen_module'):
            with tf.variable_scope(name):
                x = conv_layer(input_, out_channels, ks, s, name=name + '_l1')
                x = conv_layer(x, out_channels, ks, s, name=name + '_l2')
                x = conv_layer(x, out_channels, ks, s, name=name + '_l3')
                resid_l = input_ + x
                m_out = relu(resid_l)
                return m_out
            
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
                
            l1 = conv_layer(image, args.ngf, name='convlayer1') 
            module1 = gen_module(l1, args.ngf, name='gen_module1')
            module2 = gen_module(module1, args.ngf, name='gen_module2')
            module3 = gen_module(module2, args.ngf, name='gen_module3')
            module4 = gen_module(module3, args.ngf, name='gen_module4')
            module5 = gen_module(module4, args.ngf, name='gen_module5')
            module6 = gen_module(module5, args.ngf, name='gen_module6')
            concate_layer = tf.concat(\
                              [l1, module1, module2, module3, module4, module5, module6], \
                              axis=3, name='concat_layer')
            last_conv_layer_1 = conv_layer(concate_layer, args.ngf, ks=3, s=1, 
                                        name='last_conv_layer_1')
            last_conv_layer_2 = conv_layer(last_conv_layer_1, args.nglf, ks=3, s=1, \
                                         name='last_conv_layer_2')
            
            output = conv2d(last_conv_layer_2, args.depth, ks=3, s=1)
            output = output + image
            output = tanh(output) if args.actl == 'tanh' \
                    else sigmoid(output) if args.actl == 'sigmoid' else output
            return output


"""
def Network component
"""
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
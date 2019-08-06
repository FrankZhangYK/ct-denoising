# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu

ref : https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
"""

import os
import tensorflow as tf
import numpy as np
import time
import inout_util as ut
import pix2pix_network as nt
from tqdm import tqdm

class pix2Pix(object):
    def __init__(self, sess, args):
        self.sess = sess    

        #### set modules (generator, discriminator)
        self.g_net = nt.generator
        self.d_net = nt.discriminator
        
        """
        build model
        """                       
        assert args.phase in ['train', 'test'], 'phase : train or test'
        if args.phase=='test':
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)
            self.whole_xi, self.whole_yi = self.sample_image_loader.loader()
            self.G_whole_xi = self.g_net(args, self.whole_xi, reuse=False)

        elif args.phase=='train':
            self.image_loader = ut.DataLoader(args)
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)

            self.x_i, self.y_i = self.image_loader.loader()
            self.whole_xi, self.whole_yi = self.sample_image_loader.loader()

            #### generate & discriminate & feature extractor
            #generated images
            self.G_xi = self.g_net(args, self.x_i, reuse = False)
            self.G_whole_xi = self.g_net(args, self.whole_xi) #for sample check

            #discriminate
            self.D_xGxi= self.d_net(args, self.x_i, self.G_xi, reuse = False)
            self.D_xyi = self.d_net(args, self.x_i, self.y_i)

 
            #### loss define : L1 + cGAN
            #discriminator loss
            self.EPS = 1e-12 # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
            self.D_loss = tf.reduce_mean(-(tf.log(self.D_xyi + self.EPS) + tf.log(1 - self.D_xGxi + self.EPS)))

            #generator loss
            self.gen_loss_GAN = tf.reduce_mean(-tf.log(self.D_xGxi + self.EPS))
            self.gen_loss_L1 = tf.reduce_mean(tf.abs(self.y_i - self.G_xi))
            self.G_loss = args.gan_weight * self.gen_loss_GAN + args.l1_weight * self.gen_loss_L1

            #### variable list
            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]

            """
            summary
            """
            #loss summary
            self.summary_d_loss = tf.summary.scalar("1_DiscriminatorLoss", self.D_loss)
            self.summary_g_loss = tf.summary.scalar("2_GeneratorLoss", self.G_loss)
            self.summary_d_loss_1 = tf.summary.scalar("3_G_loss_adv", self.gen_loss_GAN)
            self.summary_d_loss_2 = tf.summary.scalar("4_G_loss_L1", self.gen_loss_L1)
            
            self.summary_all_loss = tf.summary.merge([self.summary_d_loss, self.summary_g_loss, self.summary_d_loss_1, self.summary_d_loss_2, ])
                
            #psnr summary
            self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", \
                ut.tf_psnr(self.whole_xi, self.whole_yi, self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr_result = tf.summary.scalar("2_psnr_output", \
                ut.tf_psnr(self.whole_yi, self.G_whole_xi, self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

            #image summary
            self.check_img_summary = tf.concat([tf.expand_dims(self.x_i[0], axis=0), \
                                                tf.expand_dims(self.y_i[0], axis=0), \
                                                tf.expand_dims(self.G_xi[0], axis=0)], axis = 2)        
            self.summary_train_image = tf.summary.image('0_train_image', self.check_img_summary)                                    
            self.whole_img_summary = tf.concat([self.whole_xi, self.whole_yi, self.G_whole_xi], axis = 2)        
            self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)
            
            #### optimizer
            #self.d_adam, self.g_adam = None, None
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_adam = tf.train.AdamOptimizer(\
                         learning_rate= args.lr, beta1 = args.beta1, \
                         beta2 = args.beta2).minimize(self.D_loss, var_list = self.d_vars)
                self.g_adam = tf.train.AdamOptimizer(\
                         learning_rate= args.lr, beta1 = args.beta1, \
                         beta2 = args.beta2).minimize(self.G_loss, var_list = self.g_vars)

            print('--------------------------------------------\n# of parameters : {} '.\
                  format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

                    
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    
            
    def train(self, args):
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.image_loader.tfboard_save_dir, self.sess.graph)

        self.start_step = 0
        if args.continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
        #iteration -> epoch
        self.start_epoch = \
          int((self.start_step + 1) / len(self.image_loader.data_index))
        epoch_size = len(self.image_loader.data_index)
        if args.save_freq == -1:
            args.save_freq = epoch_size
        print('Start point : iter : {}, epoch : {}, save freq : {}'.format(\
            self.start_step, self.start_epoch, args.save_freq))    
        start_time = time.time()
        
        if self.start_epoch < args.end_epoch:
            for epoch in range(self.start_epoch, args.end_epoch):
                for _ in range(0, epoch_size):
                    #discriminator update
                    _ = self.sess.run(self.d_adam)

                    #generator update & loss summary
                    _, summary_str= self.sess.run([self.g_adam, self.summary_all_loss])
                    self.writer.add_summary(summary_str, self.start_step)

                    #print point
                    if (self.start_step+1) % args.print_freq == 0:
                        #print loss & time 
                        d_loss, g_loss, g_zi_img = self.sess.run(\
                           [self.D_loss, self.G_loss, self.G_xi])
                        
                        print('Iter {} Time {} d_loss {} g_loss {}'.format(\
                                self.start_step, time.time() - start_time, d_loss, g_loss))
                        
                    if (self.start_step+1) % args.print_sample_freq == 0:
                        #training sample check
                        summary_str0 = self.sess.run(self.summary_train_image)
                        self.writer.add_summary(summary_str0, self.start_step)
                        #test sample check
                        self.check_sample(self.start_step)

                    if (self.start_step+1) % args.save_freq == 0:
                        self.save(self.start_step)
                    self.start_step += 1
            self.save(self.start_step)
        else:
            print('train complete!, trained model start epoch : {}, \
                   end_epoch :  {}'.format(self.start_epoch, self.end_epoch))

    #summary test sample image during training
    def check_sample(self, t):
        summary_str1, summary_str2 = self.sess.run(\
            [self.summary_image, self.summary_psnr])
        self.writer.add_summary(summary_str1, t)
        self.writer.add_summary(summary_str2, t)

    def save(self, step):
        model_name = "PIX2PIX.model"

        self.saver.save(self.sess,
                        os.path.join(self.image_loader.model_save_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.sample_image_loader.model_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.sample_image_loader.model_save_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

    def inference(self, args):
        self.sess.run(tf.global_variables_initializer())

        assert self.load(), 'erorr: trained model is not exsist'

        ## test
        for idx in tqdm(self.sample_image_loader.data_index):
            test_X, test_Y, output_img = self.sess.run([self.whole_xi, self.whole_yi, self.G_whole_xi])
            
            save_file_nm_g = 'Gen_from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
            np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_g), output_img)
            
            if args.raw_output:
                save_file_nm_f = 'from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
                save_file_nm_t = 'to_' +  self.sample_image_loader.target_path_list[idx].split('/')[-1][:-4]
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_f), test_X)
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_t), test_Y)

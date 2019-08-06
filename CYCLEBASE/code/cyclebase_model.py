# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from random import shuffle

import cyclebase_module as md
import cyclebase_networks as nt
import inout_util as ut
from tqdm import tqdm


class cyclebase_model(object):
    def __init__(self, sess, args):
        self.sess = sess        
        
        #### set networks
        self.discriminator = nt.discriminator
        self.generator = nt.generator
        
        """
        build model
        """
        assert args.phase in ['train', 'test'], 'phase : train or test'
        if args.phase=='test':
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)
            self.whole_A, self.whole_B = self.sample_image_loader.loader()
            self.sample_GA = self.generator(\
                args, self.whole_A, False, name="generatorA2B") #for training sample check

        elif args.phase=='train':
            self.image_loader = ut.DataLoader(args)
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)

            self.real_A, self.real_B = self.image_loader.loader()
            self.whole_A, self.whole_B = self.sample_image_loader.loader()

            #### Generator & Discriminator
            #Generator
            self.G_A = self.generator(args, self.real_A, False, name="generatorA2B")
            self.F_GA = self.generator(args, self.G_A, False, name="generatorB2A")
            self.F_B = self.generator(args, self.real_B, True, name="generatorB2A")
            self.G_FB = self.generator(args, self.F_B, True, name="generatorA2B")
            
            self.G_B = self.generator(args, self.real_B, True, \
                name="generatorA2B") #for identity loss
            self.F_A = self.generator(args, self.real_A, True, \
                name="generatorB2A") #for identity loss
                    
            self.sample_GA = self.generator(\
                    args, self.whole_A, True, name="generatorA2B") #for training sample check
            #Discriminator
            self.D_GA = self.discriminator(args, self.G_A, reuse=False, name="discriminatorB") 
            self.D_FB = self.discriminator(args, self.F_B, reuse=False, name="discriminatorA") 
            self.D_B = self.discriminator(args, self.real_B, reuse=True, name="discriminatorB")
            self.D_A = self.discriminator(args, self.real_A, reuse=True, name="discriminatorA")

            #### Losses
            # Generator loss
            self.adversarial_loss, self.Gen_loss_A2B, self.Gen_loss_B2A = \
                    md.adv_gen_loss(self.D_GA, self.D_FB)
            self.cycle_loss = \
                    md.cycle_loss(self.real_A, self.F_GA, self.real_B, self.G_FB, args.lambda_)
            self.identity_loss = \
                    md.identity_loss(self.real_A, self.F_A, self.real_B, self.G_B, args.gamma_)
            
            self.Gen_loss = self.adversarial_loss + self.cycle_loss + self.identity_loss
            
            # Dicriminator loss 
            discriminator_loss_inputs = [self.D_A, self.D_FB, \
                                         self.D_B, self.D_GA]
            self.D_loss, self.D_loss_real_A, self.D_loss_Gen_FB, \
            self.D_loss_real_B, self.D_loss_Gen_GA, \
            self.D_loss_A, self.D_loss_B = md.adv_disc_loss(discriminator_loss_inputs)
        
            #### variable list & optimizer
            if args.update_mode ==2:
                t_vars = tf.trainable_variables()
                self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
                self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
                self.g_vars = [var for var in t_vars if 'generator' in var.name]
                
                self.lr = tf.constant(args.lr, dtype=tf.float32)
                self.da_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                    .minimize(self.D_loss_A, var_list=self.da_vars)
                self.db_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                    .minimize(self.D_loss_B, var_list=self.db_vars)
                self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                    .minimize(self.Gen_loss, var_list=self.g_vars)
            else:
                t_vars = tf.trainable_variables()
                self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
                self.g_vars = [var for var in t_vars if 'generator' in var.name]

                self.lr = tf.constant(args.lr, dtype=tf.float32)
                self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                    .minimize(self.D_loss, var_list=self.d_vars)
                self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                    .minimize(self.Gen_loss, var_list=self.g_vars)
            """
            Summary
            """
            #### summary loss 
            # generator
            g_loss_summary_inputs = [self.Gen_loss, self.Gen_loss_A2B, self.Gen_loss_B2A, \
                                     self.cycle_loss, self.identity_loss, args.lambda_, args.gamma_]
            self.g_sum = md.genrator_loss_summary(g_loss_summary_inputs)

            # discriminator
            d_loss_summary_inputs = [self.D_loss, \
                                 self.D_loss_A, self.D_loss_real_A, self.D_loss_Gen_FB, \
                                 self.D_loss_B, self.D_loss_real_B, self.D_loss_Gen_GA]
            self.d_sum = md.discriminator_loss_summary(d_loss_summary_inputs)
            
            #### summary image
            img_summray_inputs = \
                self.real_A, self.real_B, self.G_A, self.whole_A, self.whole_B, self.sample_GA
            self.summary_image_1, self.summary_image_2 = \
                    md.image_summary(img_summray_inputs)

            #### summary psnr
            
            psnr_summary_inputs = \
                self.whole_A, self.whole_B, self.whole_B, self.sample_GA, self.sample_image_loader.psnr_range
            self.summary_psnr = md.psnr_summary(psnr_summary_inputs)
            
            
            #image pool
            self.pool = md.ImagePool(args.max_size)
            
            print('--------------------------------------------\n# of parameters : {} '.\
                 format(np.sum([np.prod(v.get_shape().as_list()) \
                    for v in tf.trainable_variables()])))

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)


    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.image_loader.tfboard_save_dir, self.sess.graph)

        #load SUCESS -> self.start_step 파일명에 의해 초기화... // failed -> 0

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

                #decay learning rate
                if epoch > args.decay_epoch:
                     self.lr = tf.constant(args.lr - (epoch - (args.decay_epoch)) * \
                        ((args.lr / (args.end_epoch - args.decay_epoch))), dtype=tf.float32)

                for _ in range(0, epoch_size):
                    #Update G network & generate images
                    F_B, G_A, gloss_summary_str, _ = self.sess.run(
                            [self.F_B, self.G_A, self.g_sum, self.g_optim])
                    self.writer.add_summary(gloss_summary_str, self.start_step)

                    #image pool
                    if args.img_pool:
                        [F_B, G_A] = \
                        self.pool([F_B, G_A])
                        
                    # Update D network
                    if args.update_mode ==2:
                        _, _, dloss_summary_str, lr = self.sess.run(
                            [self.da_optim, self.db_optim, self.d_sum, self.lr])
                    else:
                        _, dloss_summary_str, lr = self.sess.run(
                            [self.d_optim, self.d_sum, self.lr])
                    self.writer.add_summary(dloss_summary_str, self.start_step)

                    if (self.start_step+1) % args.print_freq == 0:
                        currt_step = self.start_step\
                                % len(self.image_loader.data_index)\
                                if epoch != 0 else self.start_step
                        print(("Epoch: {} {}/{} time: {} lr {}: ".format(\
                            epoch, currt_step, epoch_size, time.time() - start_time, lr)))

                    if (self.start_step+1) % args.print_sample_freq == 0:
                        #summary trainig sample image
                        summary_str1 = self.sess.run(self.summary_image_1)
                        self.writer.add_summary(summary_str1, self.start_step)

                        #check sample image
                        self.check_sample(self.start_step)

                    if (self.start_step+1) % args.save_freq == 0:
                        self.save(args, self.start_step)

                    self.start_step += 1
            self.save(args, self.start_step)
        else:
            print('train complete!, trained model start epoch : {}, \
                   end_epoch :  {}'.format(self.start_epoch, self.end_epoch))
        
    #summary test sample image during training
    def check_sample(self, t):
        #summary whole image'
        summary_str1, summary_str2 = self.sess.run(\
                                       [self.summary_image_2, self.summary_psnr])
        self.writer.add_summary(summary_str1, t)
        self.writer.add_summary(summary_str2, t)


    # save model    
    def save(self, args, step):
        if args.gamma_ == 0:
            model_name = "CYCLE_LOSS.model"
        else:
            model_name = "CYCLE_IDENT_LOSS.model"

        self.saver.save(self.sess,
                        os.path.join(self.image_loader.model_save_dir, model_name),
                        global_step=step)

    #load saved model
    def load(self):
        # if args.slt_model =='best_psnr':
        #     ~~~
        # else:
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
            test_A, test_B, output_img = self.sess.run([self.whole_A, self.whole_B, self.sample_GA])
            
            save_file_nm_g = 'Gen_from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
            np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_g), output_img)
            
            if args.raw_output:
                save_file_nm_f = 'from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
                save_file_nm_t = 'to_' +  self.sample_image_loader.target_path_list[idx].split('/')[-1][:-4]
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_f), test_A)
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_t), test_B)

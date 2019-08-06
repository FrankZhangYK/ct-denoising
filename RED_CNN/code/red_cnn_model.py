# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""


import os
import tensorflow as tf
import numpy as np
import time
import red_cnn_network as nt
import inout_util as ut
from tqdm import tqdm

class redCNN(object):
    def __init__(self, sess, args):
        self.sess = sess    
        
        #### set network
        self.red_cnn = nt.redcnn
        
        """
        build model
        """
        assert args.phase in ['train', 'test'], 'phase : train or test'
        if args.phase=='test':
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)
            self.whole_X, self.whole_Y = self.sample_image_loader.loader()
            self.WHOLE_output_img  = self.red_cnn(self.whole_X, reuse=False)
        
        elif args.phase=='train':
            self.image_loader = ut.DataLoader(args)
            self.X, self.Y = self.image_loader.loader()
            self.output_img = self.red_cnn(self.X, reuse = False)
            
            #### loss
            self.loss = tf.reduce_mean((self.output_img - self.Y)**2)

            #### trainable variable list
            self.t_vars = tf.trainable_variables()

            #### optimizer
            self.lr = tf.constant(args.start_lr, dtype=tf.float32)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).\
                                minimize(self.loss, var_list = self.t_vars)

            """
            summary
            """
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)
            self.whole_X, self.whole_Y = self.sample_image_loader.loader()
            self.WHOLE_output_img  = self.red_cnn(self.whole_X)
            
            #loss summary
            self.summary_loss = tf.summary.scalar("loss", self.loss)
            #psnr summary
            self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", \
                ut.tf_psnr(self.whole_Y, self.whole_X, \
                           self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr_result = tf.summary.scalar("2_psnr_output", \
                ut.tf_psnr(self.whole_Y, self.WHOLE_output_img, \
                           self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])
            
            #image summary
            self.check_img_summary = tf.concat([tf.expand_dims(self.X[0], axis=0), \
                                        tf.expand_dims(self.Y[0], axis=0), \
                                        tf.expand_dims(self.output_img[0], axis=0)], axis = 2) 
            
            self.summary_train_image = tf.summary.image('0_train_image', \
                                                        self.check_img_summary)                                    
            self.whole_img_summary = \
                tf.concat([self.whole_X, self.whole_Y, self.WHOLE_output_img], axis = 2)
            self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)
        
            print('--------------------------------------------\n# of parameters : {} '.\
                  format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))    

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    

    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.image_loader.tfboard_save_dir,\
                                            self.sess.graph)

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
        end_iter = args.end_epoch * epoch_size
        if args.save_freq == -1:
            args.save_freq = epoch_size
        print('Start point : iter : {}, epoch : {}, save freq : {}'.format(\
            self.start_step, self.start_epoch, args.save_freq))    
        start_time = time.time()
        
        
        
        if self.start_epoch < args.end_epoch:
            for epoch in range(self.start_epoch, args.end_epoch):
                for _ in range(0, epoch_size):
                    
                    self.lr = tf.constant(args.start_lr-\
                                          (args.start_lr-args.end_lr)*\
                                          (self.start_step/(end_iter-1))\
                                          , dtype=tf.float32)
                    #summary loss
                    _, summary_str= self.sess.run([self.optimizer, self.summary_loss])
                    self.writer.add_summary(summary_str, self.start_step)

                    #print point
                    if (self.start_step+1) % args.print_freq == 0:
                        loss, lr = self.sess.run([self.loss, self.lr])
                        print('Iter {} Time {} loss {}, learning_rate : {}'.\
                              format(self.start_step, time.time() - start_time, loss, lr))

                    if (self.start_step+1) % args.print_sample_freq == 0:
                        # sample check(training set)
                        summary_str0 = self.sess.run(self.summary_train_image)
                        self.writer.add_summary(summary_str0, self.start_step)
                        # sample check(test set)
                        self.check_sample(self.start_step)

                    if (self.start_step+1) % args.save_freq == 0:
                        self.save(args, self.start_step)
                    self.start_step += 1
        else:
            print('train complete!, trained model start epoch : {}, \
                   end_epoch :  {}'.format(self.start_epoch, self.end_epoch))
        
        
    #summary test sample image during training
    def check_sample(self, t):
        #summary whole image'
        summary_str1, summary_str2 = self.sess.run(\
                                       [self.summary_image, self.summary_psnr])
        self.writer.add_summary(summary_str1, t)
        self.writer.add_summary(summary_str2, t)

        
    def save(self, args, step):
        model_name = "RED_CNN.model"
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
            test_X, test_Y, output_img = self.sess.run([self.whole_X, self.whole_Y, self.WHOLE_output_img])
            
            save_file_nm_g = 'Gen_from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
            np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_g), output_img)
            
            if args.raw_output:
                save_file_nm_f = 'from_' +  self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
                save_file_nm_t = 'to_' +  self.sample_image_loader.target_path_list[idx].split('/')[-1][:-4]
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_f), test_X)
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, save_file_nm_t), test_Y)

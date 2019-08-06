# -*- coding: utf-8 -*-
"""
@author: yeohyeongyu
"""


import os
import tensorflow as tf
import numpy as np
import time
import inout_util as ut
import wgan_vgg_network as nt
from tqdm import tqdm

class wganVgg(object):
    def __init__(self, sess, args):
        self.sess = sess    

        #### set modules (generator, discriminator, vgg net)
        self.g_net = nt.generator
        self.d_net = nt.discriminator
        self.vgg = nt.Vgg19(size = args.patch_size, vgg_path = args.pretrained_vgg) 
        
        """
        build model
        """                       
        assert args.phase in ['train', 'test'], 'phase : train or test'
        if args.phase=='test':
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)
            self.whole_z, self.whole_x = self.sample_image_loader.loader()
            self.G_whole_zi = self.g_net(self.whole_z, reuse=False)

        elif args.phase=='train':
            self.image_loader = ut.DataLoader(args)
            self.sample_image_loader = ut.DataLoader(args, sample_ck=True)

            self.z_i, self.x_i = self.image_loader.loader()
            self.whole_z, self.whole_x = self.sample_image_loader.loader()

            #### generate & discriminate & feature extractor
            #generated images
            self.G_zi = self.g_net(self.z_i, reuse = False)
            self.G_whole_zi = self.g_net(self.whole_z) #for sample check

            #discriminate
            self.D_G_zi= self.d_net(self.G_zi, reuse = False)
            self.D_xi = self.d_net(self.x_i)

            #make 3-channel img for pretrained_vgg model input
            self.G_zi_3c = tf.concat([self.G_zi]*3, axis=-1)
            self.xi_3c = tf.concat([self.x_i]*3, axis=-1)
            self.E_g_zi = self.vgg.extract_feature(self.G_zi_3c)
            self.E_xi = self.vgg.extract_feature(self.xi_3c)
            [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]

            #### loss define
            #discriminator loss
            self.wgan_d_loss = -tf.reduce_mean(self.D_xi) + tf.reduce_mean(self.D_G_zi) 
            self.epsilon = tf.random_uniform([], 0.0, 1.0)
            self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
            self.D_x_hat = self.d_net(self.x_hat)
            self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]
            self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat)))
            self.grad_penal = args.lambda_ * tf.reduce_mean(tf.square(self.grad_x_hat_l2 - \
                                   tf.ones_like(self.grad_x_hat_l2)))
            
            self.D_loss = self.wgan_d_loss + self.grad_penal

            #generator loss
            self.frobenius_norm2 = tf.reduce_sum(tf.square(self.E_g_zi - self.E_xi))
            self.vgg_perc_loss = tf.reduce_mean(self.frobenius_norm2/(w*h*d))
            self.G_loss = args.lambda_1 * self.vgg_perc_loss - tf.reduce_mean(self.D_G_zi)


            #### variable list
            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]

            """
            summary
            """
            #loss summary
            self.summary_vgg_perc_loss = tf.summary.scalar("1_PerceptualLoss_VGG", \
                                                           self.vgg_perc_loss)
            self.summary_d_loss_all = tf.summary.scalar("2_DiscriminatorLoss", self.D_loss)
            self.summary_d_loss_1 = tf.summary.scalar("3_D_loss_wgan", self.wgan_d_loss)
            self.summary_d_loss_2 = tf.summary.scalar("4_D_loss_gradient_penalty", \
                                                      self.grad_penal)
            self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
            self.summary_all_loss = tf.summary.merge([self.summary_vgg_perc_loss,\
                                                      self.summary_d_loss_all,\
                                                      self.summary_d_loss_1, \
                                                      self.summary_d_loss_2, \
                                                      self.summary_g_loss])
                
            #psnr summary
            self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", \
                ut.tf_psnr(self.whole_z, self.whole_x, \
                           self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr_result = tf.summary.scalar("2_psnr_output", \
                ut.tf_psnr(self.whole_x, self.G_whole_zi, \
                           self.sample_image_loader.psnr_range), family = 'PSNR')
            self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

            #image summary
            self.check_img_summary = tf.concat([tf.expand_dims(self.z_i[0], axis=0), \
                                                tf.expand_dims(self.x_i[0], axis=0), \
                                                tf.expand_dims(self.G_zi[0], axis=0)], \
                                               axis = 2)        
            self.summary_train_image = tf.summary.image('0_train_image', \
                                                        self.check_img_summary)                                    
            self.whole_img_summary = tf.concat([self.whole_z, \
                                                self.whole_x, self.G_whole_zi], axis = 2)
            self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)
            
            #### optimizer
            #self.d_adam, self.g_adam = None, None
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_adam = tf.train.AdamOptimizer(learning_rate= args.lr, \
                                                     beta1 = args.beta1, \
                                                     beta2 = args.beta2).\
                                    minimize(self.D_loss, var_list = self.d_vars)
                self.g_adam = tf.train.AdamOptimizer(learning_rate= args.lr, \
                                                     beta1 = args.beta1, \
                                                     beta2 = args.beta2).\
                                    minimize(self.G_loss, var_list = self.g_vars)

            print('--------------------------------------------\n# of parameters : {} '.\
                  format(np.sum([np.prod(v.get_shape().as_list()) \
                                     for v in tf.trainable_variables()])))

                    
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    
            
    def train(self, args):
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.image_loader.tfboard_save_dir, \
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
        
        if args.save_freq == -1:
            args.save_freq = epoch_size
        print('Start point : iter : {}, epoch : {}, save freq : {}'.format(\
            self.start_step, self.start_epoch, args.save_freq))    
        start_time = time.time()

        if self.start_epoch < args.end_epoch:
            for epoch in range(self.start_epoch, args.end_epoch):
                for _ in range(0, epoch_size):
                    for _ in range(0, args.d_iters):
                        #discriminator update
                        _ = self.sess.run(self.d_adam)

                    #generator update & loss summary
                    _, summary_str= self.sess.run([self.g_adam, self.summary_all_loss])
                    self.writer.add_summary(summary_str, self.start_step)

                    #print point
                    if (self.start_step+1) % args.print_freq == 0:
                        currt_step = self.start_step\
                                % len(self.image_loader.data_index)\
                                if epoch != 0 else self.start_step
                        #print loss & time 
                        d_loss, g_loss = self.sess.run([self.D_loss, self.G_loss])
                        print(("Epoch: {} {}/{} time: {} d_loss {} g_loss {}".format(\
                            epoch, currt_step, epoch_size, time.time() - start_time, \
                            d_loss, g_loss)))
                        
                    if (self.start_step+1) % args.print_sample_freq == 0:
                        #training sample check
                        summary_str0 = self.sess.run(self.summary_train_image)
                        self.writer.add_summary(summary_str0, self.start_step)
                        
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
        model_name = "WGAN_VGG.model"

        self.saver.save(self.sess,
                        os.path.join(self.image_loader.model_save_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.sample_image_loader.model_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(\
                                   self.sample_image_loader.model_save_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

    def inference(self, args):
        self.sess.run(tf.global_variables_initializer())

        assert self.load(), 'erorr: trained model is not exsist'

        ## test
        for idx in tqdm(self.sample_image_loader.data_index):
            test_X, test_Y, output_img = self.sess.run(\
                                       [self.whole_z, self.whole_x, self.G_whole_zi])
            
            save_file_nm_g = 'Gen_from_' + \
                self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
            np.save(os.path.join(self.sample_image_loader.inf_save_dir, \
                                     save_file_nm_g), output_img)
            
            if args.raw_output:
                save_file_nm_f = 'from_' + \
                    self.sample_image_loader.input_path_list[idx].split('/')[-1][:-4]
                save_file_nm_t = 'to_' + \
                    self.sample_image_loader.target_path_list[idx].split('/')[-1][:-4]
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, \
                                     save_file_nm_f), test_X)
                np.save(os.path.join(self.sample_image_loader.inf_save_dir, \
                                     save_file_nm_t), test_Y)

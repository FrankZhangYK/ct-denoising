# CYCLEGAN_BASE_MODEL-tensorflow
>	* CYCLE_GAN : https://github.com/hyeongyuy/ct-denoising/tree/master/cyclegan
>	* CYCLE_IDENTITY_GAN :https://github.com/hyeongyuy/ct-denoising/tree/master/CYCLE_IDENTITY_GAN
>	* reference code:  
>     * cyclegan : https://github.com/xhujoy/CycleGAN-tensorflow

## Training detail  
> * opt : Adam(learning rate = 0.0002, beta1 = 0.5, beta2 = 0.999)
> * learning rate decay : first 100 eppoch 0.0002 and linearly decreased it to zero over the next epochs.


## Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * input_path : LDCT image folder name
> * target_path : NDCT image folder name
> * train_ratio : train <-> test (random sampling)
> * seed : random seed(train,test random sampling seed)
> * result : save result dir(check point, test, log, summary params)
> * checkpoint_dir : save directory - trained model
> * log_dir : save directory - tensoroard model
> * inference_result : save directory - test numpy file
* Image info
> * patch_size : patch size 
> * whole_size : whole size
> * depth : image channel
> * trun_max : truncated image max value
> * trun_min : truncated image min value
* Train/Test
> * phase : train | test
* Training detail
> * img_pool : use image pool (default = True)
> * strct : cyc : resnet base cyclegan(CYCLEGAN), ident : identity loss paper (default = ident)
> * augument : augumentation (default = False)
> * actl : generator last layer activation function(tanh, sigmoid, none
> * norm : normalization range, n-11 : -1 ~ 1, n01 : 0 ~ 1(default = n-11)
> * is_unpair : unpaired image (default = True)
> * max_size : image pool size (default = 50)
> * end_epoch : end epoch (default = 160)
> * decay_epoch : epoch to decay lr (default = 100)
> * lr : learning rate (default=0.0002)
> * batch_size : batch size (default=10)
> * lambda_ : weight of cyclic loss (default=10)
> * gamma_ : weight of identity loss (default=5)
> * beta1 : Adam optimizer parameter (default=0.5)
> * beta2 : Adam optimizer parameter (default=0.999)
> * ngf : # of generator filters in first conv layer
> * nglf : # of generator filters in last conv layer
> * ndf : # of discriminator filters in first conv layer

* others
> * save_freq : save a model every step (iterations)
> * print_freq : print frequency (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)
> * prefetch_buffer : prefetch buffer size 
> * num_parallel_calls : num parallel calls
> * raw_output : True : outpu raw image

# ct-denoising
## Denoising Model
* RED_CNN
>	* paper :https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf
* WGAN_VGG
>	* paper : https://arxiv.org/pdf/1708.00961.pdf
>	* original code:  
>     * vgg : https://github.com/machrisaa/tensorflow-vgg  
>     * WGAN : https://github.com/jiamings/wgan
* CYCLEGAN
>	* paper : https://arxiv.org/abs/1703.10593
>	* original code: https://github.com/xhujoy/CycleGAN-tensorflow

## Run command example
$ touch run_cycleident.sh <br>
$ vi run_cycleident.sh <br>
> nohup python main.py --end_epoch=160 --decay_epoch=100 --result=/data/hyeongyu/CYCLE_IDENT --phase=train > cycle_ident_train_log --gpu_no=3 && <br>
> nohup python main.py --result=/data/hyeongyu/CYCLE_IDENT --phase=test > cycle_ident_test_log --gpu_no=0 & <br>

$ sh run_cycleident.sh
## Output (EX cycle_ident)
1. tensorboard
![cycle_ident_tsbd_ex](https://github.com/hyeongyuy/ct-denoising/blob/master/img/cycle_ident_tsbd_ex.jpg)
2. parameter_summary.txt
> train complete!!, <br> 
time : ~~(s) <br>
set params : <br>
Namespace(actl='tanh', augument=False, batch_size=10, beta1=0.5, beta2=0.999, checkpoint_dir='model', continue_train=True, dcm_path='/data/DICOM', decay_epoch=100, depth=1, end_epoch=160, extension='DCM', gamma_=5.0, gpu_no=3, img_pool=True, inference_result='output', input_path='LDCT', is_unpair=True, lambda_=10.0, log_dir='logs', lr=0.0002, max_size=50, ndf=64, ngf=128, nglf=15, norm='n-11', num_parallel_calls=6, patch_size=56, phase='train', prefetch_buffer=100, print_freq=500, print_sample_freq=2000, raw_output=False, result='/data1/hyeongyu/EXP_RESULTS/CYCLE_IDENT', save_freq=~~~, seed=0, slt_model='latest', strct='ident', target_path='NDCT', train_ratio=0.7, trun_max=2000, trun_min=-1000, update_mode=2, whole_size=512)
3. model
4. generated img(.npy)

## [Common] Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * input_path : LDCT image folder name
> * target_path : NDCT image folder name
> * train_ratio : train <-> test ratio (random sampling)
> * result : save result dir(check point, test, log, summary params)
> * checkpoint_dir : save directory - trained model
> * log_dir : save directory - tensoroard model
> * inference_result : save directory - test numpy file
* Image info
> * patch_size : patch size 
> * whole_size : whole size
> * depth : input image channel
> * trun_max : truncated image max value
> * trun_min : truncated image min value
* Train/Test
> * phase : train | test
* others
> * save_freq : save a model every save_freq (iterations)
> * print_freq : print_freq (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)

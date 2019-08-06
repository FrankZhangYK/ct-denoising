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
$ touch run_cycleident.sh
$ vi run_cycleident
> nohup python main.py --end_epoch=10 --decay_epoch=5 --result=/data/hyeongyu/CYCLE_IDENT --phase=train > cycle_ident_train_log --gpu_no=0 &&
> nohup python main.py --result=/data/hyeongyu/CYCLE_IDENT --phase=test > cycle_ident_test_log --gpu_no=0 &

## Output
1. tensorboard
2. input params
3. model
4. generated img(.npy)

## [Common] Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * LDCT_path : LDCT image folder name
> * NDCT_path : NDCT image folder name
> * test_patient_no : test patient id list(p_id1,p_id2...) (train patient id : (patient id list - test patient id list)
> * result : save result dir(check point, test, log, summary params)
> * checkpoint_dir : save directory - trained model
> * log_dir : save directory - tensoroard model
> * test_npy_save_dir : save directory - test numpy file
> * pretrained_vgg : pretrained vggnet directory(only WGAN_VGG)
* Image info
> * patch_size : patch size (WGAN_VGG, RED_CNN)
> * whole_size : whole size
> * img_channel : image channel
> * img_vmax : max value
> * img_vmin : min value
* Train/Test
> * phase : train | test
* others
> * is_mayo : summary ROI sample1,2
> * save_freq : save a model every save_freq (iterations)
> * print_freq : print_freq (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)

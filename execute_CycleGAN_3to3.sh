SAVE_ROOT_CYCLEGAN=/home/compu/SJJ/CycleGAN_seg_2.5d/cyclegan_result_0925_3to3
GPU=2
#1. train CycleGAN Network
CUDA_VISIBLE_DEVICES=${GPU} python3 train_CycleGAN_3to3.py \
  --dataset 'cardiac' \
  --batch_size 3 \
  --max_epochs 10 \
  --lr 0.0001 \
  --beta1 0 \
  --img_size 256 \
  --save_dir ${SAVE_ROOT_CYCLEGAN} \
  --n_channels 3 \
  --save_img 5 \
  --save_model 50

  #의문점...
  #
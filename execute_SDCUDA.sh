SAVE_ROOT_CYCLEGAN=/home/compu/SJJ/CycleGAN_seg_2.5d/SDCUDA_result_0911
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python3 train_SDCUDA.py \
  --dataset 'cardiac' \
  --batch_size 1 \
  --max_epochs 30 \
  --lr 0.0001 \
  --beta1 0 \
  --img_size 256 \
  --save_dir ${SAVE_ROOT_CYCLEGAN} \
  --n_channels 1\
  --save_img 50 \
  --save_model 500
SAVE_ROOT_eval=/home/compu/SJJ/CycleGAN_seg_2.5d/eval_0925_it10000
GPU=2
TRAINED_WEIGHT_SEG=/home/compu/SJJ/CycleGAN_seg_2.5d/seg_result_0925/it0080.pt
CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
  --dataset 'cardiac' \
  --batch_size 4 \
  --img_size 256 \
  --n_channels 3\
  --save_dir ${SAVE_ROOT_eval}\
  --weights ${TRAINED_WEIGHT_SEG}
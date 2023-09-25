#2. inference domain adaptation by CycleGAN and train segmentation network
GPU=2
SAVE_ROOT_SEG=/home/compu/SJJ/CycleGAN_seg_2.5d/seg_result_0925
TRAINED_WEIGHT="/home/compu/SJJ/CycleGAN_seg_2.5d/cyclegan_result_0925/it0015.pt"
if [ -f "$TRAINED_WEIGHT" ]; then
    echo "CycleGAN 학습이 완료되었습니다. TRAINED_WEIGHT: $TRAINED_WEIGHT"

    CUDA_VISIBLE_DEVICES=${GPU} python3 train_segmentation.py \
        --dataset 'cardiac' \
        --batch_size 14 \
        --max_epochs 100 \
        --lr 1e-5 \
        --weight_decay 1e-8 \
        --momentum 0.999 \
        --num_classes 5 \
        --img_size 256 \
        --save_dir ${SAVE_ROOT_SEG} \
        --weights ${TRAINED_WEIGHT} \
        --n_channels 1 \
        --save_img 5 \
        --save_model 5
else
    echo "CycleGAN 학습이 완료되지 않았거나, 올바른 경로를 설정해주세요."
fi
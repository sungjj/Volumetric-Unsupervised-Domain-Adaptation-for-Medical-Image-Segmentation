#2. inference domain adaptation by CycleGAN and train segmentation network
GPU=2
SAVE_ROOT_SEG=/home/compu/SJJ/CycleGAN_seg_2.5d/seg_result_0925_3to3
TRAINED_WEIGHT="/home/compu/SJJ/CycleGAN_seg_2.5d/cyclegan_result_0925_3to3/it0050.pt"
if [ -f "$TRAINED_WEIGHT" ]; then
    echo "CycleGAN 학습이 완료되었습니다. TRAINED_WEIGHT: $TRAINED_WEIGHT"

    CUDA_VISIBLE_DEVICES=${GPU} python3 train_segmentation_3to3.py \
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
        --n_channels 3 \
        --save_img 5 \
        --save_model 5
else
    echo "CycleGAN 학습이 완료되지 않았거나, 올바른 경로를 설정해주세요."
fi

#3. make segmentation mask
# 주의점 : segmentation mask와 image를 짝지어야 됨 ->이미지 이름까지 저장
# 2번에서 학습한 네트워크를 통과시키고 결과물을 저장함




#4. Train segmantion network using Real & Generated Target
# 3번에서 저장한 결과물, 원래 이미지 불러와서 U-Net학습


# SAVE_ROOT_SEG_EVAL=/home/compu/SJJ/CycleGAN_seg_2d/seg_eval_result


#5. evaluate Target segmentation
#4번에서 학습한 네트워크 불러와서 Target segmentation dice metric 평가
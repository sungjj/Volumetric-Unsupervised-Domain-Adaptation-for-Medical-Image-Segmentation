# Volumetric-Unsupervised-Domain-Adaptation-for-Medical-Image-Segmentation

[SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation (CVPR 2023) by Hyungseob Shin∗, Hyeongyu Kim∗, Sewon Kim, Yohan Jun, Taejoon Eo and Dosik Hwang.](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_SDC-UDA_Volumetric_Unsupervised_Domain_Adaptation_Framework_for_Slice-Direction_Continuous_Cross-Modality_CVPR_2023_paper.pdf)

This project aims to implement the core methods outlined in the aforementioned paper and compare them with existing approaches.
The following three methods were used in domain adaptation.
1. 2d CycleGAN
2. 3to3 CycleGAN
3. SDC-UDA

The training consists of two steps, 1) Image translation and 2) segmentation

## Dataset
[Cardiac (MMWHS)](https://zmiclab.github.io/zxh/0/mmwhs/)

## Segmentation Result (Dice score)
w/o Domain Adaptation : 0.0491 
2d CycleGAN: 0.5024
3to3 CycleGAN:0.4386 
SDC-UDA: 0.5929

## Domain Adaption Result
CT -> MR -> CT
![image](https://github.com/sungjj/Volumetric-Unsupervised-Domain-Adaptation-for-Medical-Image-Segmentation/assets/136042172/a3034672-9631-431a-8d67-4c5f331a60fd)

MR -> CT -> MR
![image](https://github.com/sungjj/Volumetric-Unsupervised-Domain-Adaptation-for-Medical-Image-Segmentation/assets/136042172/0d034bcc-18cc-4921-aecb-0f9b4f847320)


## Reference

The segmentation model is [UNet](https://github.com/milesial/Pytorch-UNet/tree/master)

The dataset code is partially borrowed by [DRANet](https://github.com/Seung-Hun-Lee/DRANet)

The transformer block code is partially borrowed by [Segformer](https://github.com/lucidrains/segformer-pytorch)


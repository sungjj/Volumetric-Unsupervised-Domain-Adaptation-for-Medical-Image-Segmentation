import os
import os.path as osp
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from unet_model import UNet

from data_pre import get_dataset_cardiac, get_dataset_cardiac_2d
from dice import dice_loss, multiclass_dice_coeff

def transpose(ndarray):
    return np.transpose(ndarray, [0,2,3,1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cardiac', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--n_channels", default=1, type=int)
    parser.add_argument("--save_dir", default='/home/compu/SJJ/CycleGAN_seg_2d/seg_result_test', help='directory to save images', type=str)
    parser.add_argument("--weights", default='/home/compu/SJJ/CycleGAN_seg_2d/cyclegan_result_0709_gan2_x/final_weights.pt',type=str )
    args = parser.parse_args()

    if args.dataset == 'cardiac':
        if args.n_channels==3:
            train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac((args.img_size,args.img_size), batch_size=args.batch_size)
        elif args.n_channels==1:
            train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac_2d((args.img_size,args.img_size), batch_size=args.batch_size)
    else :
        train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac_2d((args.img_size,args.img_size), batch_size=args.batch_size)
    
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    model_seg = UNet(n_channels=args.n_channels, n_classes=args.num_classes)
    model_seg.cuda()
    state_dict = torch.load(args.weights)
    model_seg.load_state_dict(state_dict['model'])
    
    num_iter=0
    running_loss=0
    max_iter = len(train_ct_loader)
    train_start_time = time.time()
    with torch.no_grad():
        sum_dice_score=0
        mean_dice_score=0
        num_iter=0
        mean=0
        model_seg.eval()
        for input_batch, label_batch in train_ct_loader:
            sum_dice_score_batch=0
            data_X= input_batch.cuda().float()
            data_X=(data_X+1)/2
            #print(data_X.size())
            output=model_seg(data_X)
            label_batch=label_batch.cuda().long()
            # gen_G_Y = model_G_X(data_X)
            # gen_G_Y=(gen_G_Y+1)/2
            # output=model_seg(gen_G_Y)
            label_batch=label_batch.squeeze(1)
            #label_batch=label_batch[:,:,:,1]
            mask_true = F.one_hot(label_batch, model_seg.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(output.argmax(dim=1), model_seg.n_classes).permute(0, 3, 1, 2).float()
            mean_dice_score_batch = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            num_iter+=1
            sum_dice_score+=mean_dice_score_batch
            mean+=mean_dice_score_batch
            if num_iter%5 == 0:
                print('it[{:04d}/{:04d}] \tdice_score:{:.4f}'.format(
                    num_iter, len(train_ct_loader), mean/5
                ))
                mean=0
                
                
            if num_iter%5 == 0:    
                save_name = osp.join(save_dir, 'it{:04d}.pt'.format(num_iter))
                image_name = 'it{:04d}.jpg'.format(num_iter)
                save_path = os.path.join(save_dir, image_name)
                
                images = data_X.detach().cpu().numpy()
                output_images = output.detach().cpu().numpy()
                label_images = label_batch.detach().cpu().numpy()

                # Just save the first image in the batch for visualization
                img = images[0, 0,:, :]
                #print(output_images.shape)
                output_img = output_images[0, :, :]
                #print(output_img.shape)
                output_img = np.argmax(output_img, axis=0)
                #print(label_images.shape)
                label_img = label_images[0, :, :]#배치 데이터 중 하나만 가져가는 상황
                #print('label:',label_img.shape)
                # label_img = label_img.astype(np.uint8)  # 값의 범위를 0에서 255로 변환
                # label_cmap = plt.cm.get_cmap('viridis', model.n_classes)  # 색상 맵 설정
                # label_img_rgb = label_cmap(label_img)[:, :, :3]  # RGB 형태로 변환

                # Plot and save
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                ax[0].imshow(img)
                ax[0].title.set_text('Generated Image')
                ax[1].imshow(output_img)
                ax[1].title.set_text('Output Image')
                ax[2].imshow(label_img)
                ax[2].title.set_text('Label Image')

                # 이미지 저장
                plt.savefig(os.path.join(save_dir, f'output_vs_label_iter{num_iter}.png'))

                # close plot
                plt.close(fig)
                
        print('mean dice score:', sum_dice_score/len(train_ct_loader))
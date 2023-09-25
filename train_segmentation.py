import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from unet_model import UNet
import torch.optim as optim

from data_pre import get_dataset_cardiac, get_dataset_cardiac_2d
from GAN_model import Generator_A2B
from dice import dice_loss

import warnings
warnings.filterwarnings("ignore")

def transpose(ndarray):
    return np.transpose(ndarray, [0,2,3,1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cardiac', type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--momentum", default=0.999, type=float)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--save_dir", default='/home/compu/SJJ/CycleGAN_seg_2d/cyclegan_result', help='directory to save images', type=str)
    parser.add_argument("--weights", default='/home/compu/SJJ/CycleGAN_seg_2d/cyclegan_result/it0050.pt',type=str )
    parser.add_argument("--n_channels", default=1,type=int )
    parser.add_argument("--save_img", default=50,help='Set the interval for saving images in terms of the number of iterations',type=int )
    parser.add_argument("--save_model", default=500,help='Set the interval for saving model parameters in terms of the number of iterations',type=int )

    args = parser.parse_args()
    if args.dataset == 'cardiac':
        train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac_2d((args.img_size,args.img_size), batch_size=args.batch_size)
    else :
        train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac_2d((args.img_size,args.img_size),batch_size=args.batch_size)
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    
    model_G_X = Generator_A2B(args.n_channels)
    model_seg = UNet(n_channels=args.n_channels, n_classes=args.num_classes)
    model_seg.cuda()
    

    criterion_L1 = nn.L1Loss()
    criterion_seg = nn.CrossEntropyLoss() if model_seg.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    optimizer_seg = optim.RMSprop(model_seg.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    model_G_X.cuda()
    state_dict=torch.load(args.weights)
    model_G_X.load_state_dict(state_dict['model_G_X'])
    model_G_X.eval()
    
    num_iter=0
    running_loss=0
    max_iter = args.max_epochs*len(train_mr_loader)
    train_start_time = time.time()
    for epoch in range(1, args.max_epochs+1):
        model_seg.train()
        for input_batch, label_batch in train_mr_loader:
            data_X= input_batch.cuda().float()
            label_batch=label_batch.cuda().long()
            gen_G_Y = model_G_X(data_X)
            gen_G_Y=(gen_G_Y+1)/2
            output=model_seg(gen_G_Y)
            label_batch=label_batch.squeeze(1)
            loss_seg=criterion_seg(output, label_batch)
            loss_seg += dice_loss(F.softmax(output, dim=1).float(),
                              F.one_hot(label_batch, model_seg.n_classes).permute(0, 3, 1, 2).float(),
                              multiclass=True
            )
            optimizer_seg.zero_grad()
            loss_seg.backward()
            running_loss += loss_seg.item()
            optimizer_seg.step()
            num_iter+=1 
            if (num_iter%args.save_model==0): 
                save_name = osp.join(save_dir, 'it{:04d}.pt'.format(num_iter))
                torch.save({'model': model_seg.state_dict()}, save_name)

                        
            if num_iter%5 == 0:
                print('it[{:04d}/{:04d}] \tLoss:{:.4f}  \telapsed_time:{:.2f}mins'.format(
                    num_iter, max_iter, running_loss/5,  (time.time()-train_start_time)/60
                ))
                running_loss=0
                
            if num_iter%args.save_img == 0:    
                save_name = osp.join(save_dir, 'it{:04d}.pt'.format(num_iter))
                image_name = 'it{:04d}.jpg'.format(num_iter)
                save_path = os.path.join(save_dir, image_name)
                
                images = gen_G_Y.detach().cpu().numpy()
                output_images = output.detach().cpu().numpy()
                label_images = label_batch.detach().cpu().numpy()

                # Just save the first image in the batch for visualization
                img = images[0, 0,:, :]
            
                output_img = output_images[0, :, :]
                output_img = np.argmax(output_img, axis=0)
                label_img = label_images[0, :, :]


                # Plot and save
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                ax[0].imshow(img)
                ax[0].title.set_text('Generated Image')
                ax[1].imshow(output_img)
                ax[1].title.set_text('Output Image')
                ax[2].imshow(label_img)
                ax[2].title.set_text('Label Image')

                # 이미지 저장
                plt.savefig(os.path.join(save_dir, f'output_vs_label_epoch{epoch}_iter{num_iter}.png'))

                # close plot
                plt.close(fig)
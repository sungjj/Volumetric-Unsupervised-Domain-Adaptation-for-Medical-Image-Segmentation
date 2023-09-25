import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from data_pre import get_dataset_cardiac_2d
from GAN_model import Generator_A2B, Generator_B2A, Discriminator

import warnings
warnings.filterwarnings("ignore")
def transpose(ndarray):
    return np.transpose(ndarray, [0,2,3,1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cardiac', type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--beta1", default=0, help='Beta1 hyperparameters for Adam optimizers', type=float)
    parser.add_argument("--img_size", default=256, help='image size of data image', type=int)
    parser.add_argument("--save_dir", default='/home/compu/SJJ/CycleGAN_seg_2d/cyclegan_result', help='directory to save images', type=str)
    parser.add_argument("--n_channels", default=1, type=int)
    parser.add_argument("--save_img", default=50,help='Set the interval for saving images in terms of the number of iterations',type=int )
    parser.add_argument("--save_model", default=500,help='Set the interval for saving model parameters in terms of the number of iterations',type=int )
    args = parser.parse_args()
    
    if args.dataset == 'cardiac':
        train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac_2d((args.img_size,args.img_size), batch_size=args.batch_size)
    else :
        train_A_loader, train_B_loader, test_A_loader, test_B_loader = get_dataset_cardiac_2d((args.img_size,args.img_size),batch_size=args.batch_size)
    
    

    model_G_X = Generator_A2B(n_channels=args.n_channels).cuda()
    model_G_Y = Generator_B2A(n_channels=args.n_channels).cuda()
    model_D_X = Discriminator(n_channels=args.n_channels).cuda()
    model_D_Y = Discriminator(n_channels=args.n_channels).cuda()

    
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    g_params = list([])
    g_params = g_params + list(model_G_X.parameters())   
    g_params = g_params + list(model_G_Y.parameters())
    optimizer_G =  torch.optim.Adam(g_params, lr=args.lr, betas=(args.beta1, 0.999))
    
    d_params = list([])
    d_params = d_params + list(model_D_X.parameters())
    d_params = d_params + list(model_D_Y.parameters())
    optimizer_D =  torch.optim.Adam(d_params, lr=args.lr, betas=(args.beta1, 0.999))
    
    
    real_label = 1
    fake_label = 0 
    label_real = torch.full((args.batch_size,args.n_channels, args.img_size, args.img_size), real_label, dtype=torch.float32, device='cuda')
    label_fake = torch.full((args.batch_size,args.n_channels ,args.img_size, args.img_size), fake_label, dtype=torch.float32, device='cuda')
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    num_iter=0
    max_iter=args.max_epochs*len(train_mr_loader)
    train_start_time=time.time()
    for epoch in range(1, args.max_epochs+1):
        for input_batch, label_batch in train_mr_loader:
            #print(input_batch.size())
            model_G_X.train()
            model_G_Y.train()
            model_D_X.train()
            model_D_Y.train()
            data_X= input_batch.cuda().float()
            data_Y, labels_Y = next(iter(train_ct_loader))
            data_Y = data_Y.cuda().float()
            model_D_X.zero_grad()
            model_D_Y.zero_grad()

            #discrimination loss

            #loss for D_X
            out_D_X_real = model_D_X(data_X)
            label_real = torch.full_like(out_D_X_real, real_label, dtype=torch.float32, device='cuda')
            err_D_X_real = criterion_GAN(out_D_X_real, label_real)
            gen_G_X = model_G_Y(data_Y)
            out_D_X_fake = model_D_X(gen_G_X.detach())#G(Y)의 gradient flow 차단
            label_fake = torch.full_like(out_D_X_fake, fake_label, dtype=torch.float32, device='cuda')
            err_D_X_fake = criterion_GAN(out_D_X_fake, label_fake)
            err_D_X = err_D_X_real + err_D_X_fake

            #loss for D_Y
            out_D_Y_real = model_D_Y(data_Y)
            label_real = torch.full_like(out_D_Y_real, real_label, dtype=torch.float32, device='cuda')
            err_D_Y_real = criterion_GAN(out_D_Y_real, label_real)
            gen_G_Y = model_G_X(data_X)
            out_D_Y_fake = model_D_Y(gen_G_Y.detach())#G(X)의 gradient flow 차단
            label_fake = torch.full_like(out_D_Y_fake, fake_label, dtype=torch.float32, device='cuda')
            err_D_Y_fake = criterion_GAN(out_D_Y_fake, label_fake)
            err_D_Y = err_D_Y_real + err_D_Y_fake

            err_D = err_D_X + err_D_Y

            err_D_X.backward()
            err_D_Y.backward()
            optimizer_D.step()

            #############


            ### Update generator model

            #############
            model_G_X.zero_grad()
            model_G_Y.zero_grad()
            
            #identity loss
            err_identity_Y=criterion_L1(model_G_X(data_Y),data_Y)
            err_identity_X=criterion_L1(model_G_Y(data_X),data_X)
            err_I=err_identity_Y+err_identity_X
            

            #generation loss
            out_D_G_X = model_D_X(model_G_Y(data_Y))
            label_real = torch.full_like(out_D_G_X, real_label, dtype=torch.float32, device='cuda')
            err_G_X = criterion_GAN(out_D_G_X, label_real)
            out_D_G_Y = model_D_Y(model_G_X(data_X))
            label_real = torch.full_like(out_D_G_Y, real_label, dtype=torch.float32, device='cuda')
            err_G_Y = criterion_GAN(out_D_G_Y, label_real)
            err_G = err_G_X + err_G_Y

            #cycle loss
            cycle_X = model_G_Y(gen_G_Y)
            cycle_Y = model_G_X(gen_G_X)

            err_cycle_X = criterion_L1(cycle_X, data_X)
            err_cycle_Y = criterion_L1(cycle_Y, data_Y)
            #print('err_cycle_X:',err_cycle_X)
            #print('err_cycle_Y:',err_cycle_Y)
            err_C = err_cycle_X + err_cycle_Y

            err_GC = err_G + 10*err_C + 5*err_I
            err_GC.backward()

            optimizer_G.step()



            #############
            num_iter += 1
            # Output training stats
            if num_iter%5 == 0:
                print('it[{:04d}/{:04d}] \tLoss_D:{:.4f} \tLoss_G:{:.4f} \tLoss_I:{:.4f} \tLoss_C:{:.4f} \telapsed_time:{:.2f}mins'.format(
                    num_iter, max_iter, err_D.item(), err_G.item(), err_I.item(), err_C.item(), (time.time()-train_start_time)/60
                ))

            if num_iter%args.save_img==0 or num_iter==max_iter:
                save_name = osp.join(save_dir, 'it{:04d}.pt'.format(num_iter))
                image_name = 'it{:04d}.jpg'.format(num_iter)
                save_path = os.path.join(save_dir, image_name)
                save_name_w=osp.join(save_dir, 'final_weights.pt')

                #plt.imshow(outputs)
                
                if num_iter%args.save_model==0 or num_iter==max_iter:
                    torch.save({
                        'model_G_X': model_G_X.state_dict(),
                        'model_G_Y': model_G_Y.state_dict()
                    }, save_name)
                if num_iter%args.save_img==0:
                    with torch.no_grad():
                        model_G_X.eval()
                        model_G_Y.eval()
                        for i, (input_batch, label_batch) in enumerate(test_mr_loader):
                            if i == 0:
                                data_X= input_batch.cuda().float()
                                #print("4:",data_X.size())
                                
                                data_Y, labels_Y = next(iter(test_ct_loader))
                                data_Y = data_Y.cuda().float()

                                output_X = model_G_X(data_X)
                                output_Y = model_G_Y(data_Y)
                                data_X=data_X.cpu().data.numpy()
                                data_X=(data_X+1)/2
                                data_Y=data_Y.cpu().data.numpy()
                                data_Y=(data_Y+1)/2
                                output_X=output_X.cpu().data.numpy()
                                output_X=(output_X+1)/2
                                output_Y=output_Y.cpu().data.numpy()
                                output_Y=(output_Y+1)/2
                                vis_num=2
                                for vis_idx in range(vis_num):
                                    data_X_, data_Y_ = transpose(data_X)[vis_idx], transpose(data_Y)[vis_idx]
                                    output_X_, output_Y_  = transpose(output_X)[vis_idx], transpose(output_Y)[vis_idx]
                                    outputs = np.concatenate((data_X_, output_X_, data_Y_, output_Y_), axis=1)
                                    plt.imshow(outputs)
                                    plt.savefig(save_path)
                                    plt.close()
                                    plt.pause(0.001)
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
from torch import autograd

from data_pre import get_dataset_cardiac
from GAN_model import Discriminator
from sdc_uda_model import SDC_UDA

def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def requires_grads_True(models):
    for model in models :     
        model.train()
        for p in model.parameters():
            p.requires_grad = True

def requires_grads_False(models):
    for model in models :     
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

def transpose(ndarray):
    return np.transpose(ndarray, [0,2,3,1])

def accumulate(model1, model2, decay=0.999):
    
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cardiac', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
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
        train_ct_loader, train_mr_loader, test_ct_loader, test_mr_loader = get_dataset_cardiac((args.img_size,args.img_size), batch_size=args.batch_size)
    else :
        train_A_loader, train_B_loader, test_A_loader, test_B_loader = get_dataset_cardiac((args.img_size,args.img_size),batch_size=args.batch_size)
    
    


    model_G_X = SDC_UDA(input_nc=args.n_channels, output_nc=args.n_channels)
    model_G_Y = SDC_UDA(input_nc=args.n_channels, output_nc=args.n_channels)
    model_D_X = Discriminator(n_channels=args.n_channels)
    model_D_Y = Discriminator(n_channels=args.n_channels)
    model_G_X_ema = SDC_UDA(input_nc=args.n_channels, output_nc=args.n_channels)
    model_G_Y_ema = SDC_UDA(input_nc=args.n_channels, output_nc=args.n_channels)
   
    
    model_G_X=model_G_X.cuda()
    model_G_Y=model_G_Y.cuda()
    model_D_X=model_D_X.cuda()
    model_D_Y=model_D_Y.cuda()
    model_G_X_ema=model_G_X_ema.cuda()
    model_G_Y_ema=model_G_Y_ema.cuda()
    
    model_G_X_ema.eval()
    model_G_Y_ema.eval()

    accumulate(model_G_X_ema, model_G_X, 0)
    accumulate(model_G_Y_ema, model_G_Y, 0)
    
    
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    d_reg_ratio= 12/(13)
    
    g_params = list([])
    g_params = g_params + list(model_G_X.parameters())   
    g_params = g_params + list(model_G_Y.parameters())
    optimizer_G =  torch.optim.Adam(g_params, lr=args.lr, betas=(args.beta1, 0.99))
    
    d_params = list([])
    d_params = d_params + list(model_D_X.parameters())
    d_params = d_params + list(model_D_Y.parameters())
    optimizer_D =  torch.optim.Adam(d_params, lr=args.lr*d_reg_ratio, betas=(args.beta1**d_reg_ratio, 0.99**d_reg_ratio))
    
    real_label = 1
    fake_label = 0 
    label_real = torch.full((args.batch_size,args.n_channels, args.img_size, args.img_size), real_label, dtype=torch.float32, device='cuda')
    label_fake = torch.full((args.batch_size,args.n_channels ,args.img_size, args.img_size), fake_label, dtype=torch.float32, device='cuda')
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    num_iter=0
    max_iter=args.max_epochs*len(train_mr_loader)
    train_start_time=time.time()
    accum = 0.5 ** (32 / (10 * 1000))
    for epoch in range(1, args.max_epochs+1):
        for input_batch, label_batch in train_mr_loader:
            #print(input_batch.size())
            
            data_X= input_batch.cuda().float()
            data_Y, labels_Y = next(iter(train_ct_loader))
            data_Y = data_Y.cuda().float()


            #############


            ### Update generator model

            #############

            requires_grads_True([ model_G_X, model_G_Y])
            requires_grads_False( [model_D_X, model_D_Y])
            #identity loss
            
            err_identity_Y=criterion_L1(model_G_X(data_Y),data_Y)
            err_identity_X=criterion_L1(model_G_Y(data_X),data_X)
            err_I=err_identity_Y+err_identity_X
            

            #generation loss
            
            fake_G_X=model_G_Y(data_Y)
            out_D_G_X = model_D_X(fake_G_X)
            err_G_X= g_nonsaturating_loss(out_D_G_X[:,0])

           
            fake_G_Y=model_G_X(data_X)
            out_D_G_Y = model_D_Y(fake_G_Y)
            err_G_Y= g_nonsaturating_loss(out_D_G_Y[:,0])
           
            err_G = err_G_X + err_G_Y

            #cycle loss
            cycle_X = model_G_Y(fake_G_Y) #X>Y>X
            cycle_Y = model_G_X(fake_G_X) #Y>X>Y

            err_cycle_X = criterion_L1(cycle_X, data_X)
            err_cycle_Y = criterion_L1(cycle_Y, data_Y)
            #print('err_cycle_X:',err_cycle_X)
            #print('err_cycle_Y:',err_cycle_Y)
            err_C = err_cycle_X + err_cycle_Y
            err_GC = 1*err_G + 3*err_C +1*err_I
            optimizer_G.zero_grad()
            err_GC.backward()
            optimizer_G.step()
            accumulate(model_G_X_ema, model_G_X, accum)
            accumulate(model_G_Y_ema, model_G_Y, accum)

            ###################################################################
            #discrimination loss
            requires_grads_True([ model_D_X, model_D_Y])
            requires_grads_False( [model_G_X, model_G_Y])
            #loss for D_X
            out_D_X_real = model_D_X(data_X)
            out_D_X_fake = model_D_X(fake_G_X.detach())
            err_D_X = d_logistic_loss(out_D_X_real[:,0], out_D_X_fake[:,0])


            #label_real = torch.full_like(out_D_X_real, real_label, dtype=torch.float32, device='cuda')
            #print(out_D_X_real.shape,'out_D_X_real.shape')
            #print(label_real.shape,'label_real.shape')

            #err_D_X_real = criterion_GAN(out_D_X_real, label_real)
            #gen_G_X = model_G_Y(data_Y)
            #G(Y)의 gradient flow 차단
            #label_fake = torch.full_like(out_D_X_fake, fake_label, dtype=torch.float32, device='cuda')
            
            
            #err_D_X_fake = criterion_GAN(out_D_X_fake, label_fake)
            #err_D_X = err_D_X_real + err_D_X_fake

            #loss for D_Y
            
            out_D_Y_real = model_D_Y(data_Y)
            out_D_Y_fake = model_D_Y(fake_G_Y.detach())
            # print(out_D_Y_real,'out_D_Y_real')
            # print(out_D_Y_fake,'out_D_Y_fake')
            # print(out_D_Y_real[:,0],'out_D_Y_real[:,0]')
            # print(out_D_Y_fake[:,0],'out_D_Y_fake[:,0]')
            err_D_Y = d_logistic_loss(out_D_Y_real[:,0], out_D_Y_fake[:,0])

            # out_D_Y_real = model_D_Y(data_Y)
            # label_real = torch.full_like(out_D_Y_real, real_label, dtype=torch.float32, device='cuda')
            

            
            # err_D_Y_real = criterion_GAN(out_D_Y_real, label_real)
            # gen_G_Y = model_G_X(data_X)
            # out_D_Y_fake = model_D_Y(gen_G_Y.detach())#G(X)의 gradient flow 차단
            # label_fake = torch.full_like(out_D_Y_fake, fake_label, dtype=torch.float32, device='cuda')
            
            
            # err_D_Y_fake = criterion_GAN(out_D_Y_fake, label_fake)
            # err_D_Y = err_D_Y_real + err_D_Y_fake

            err_D = err_D_X + err_D_Y
            print(err_D)
            optimizer_D.zero_grad()
            err_D.backward()
            optimizer_D.step()
            
            
            
            d_regularize = num_iter %12 == 0
            r1_loss = torch.tensor(0, dtype = torch.float, device='cuda')
            
            if d_regularize:
                data_X.requires_grad = True
                data_Y.requires_grad = True
            
                
                
                pred_real = model_D_X(data_X)
                r1_loss += d_r1_loss(pred_real[:,0], data_X)
                
                pred_real = model_D_Y(data_Y)
                r1_loss += d_r1_loss(pred_real[:,0], data_Y)
                
                
                optimizer_D.zero_grad()
                r1_loss_sum =  10 / 2 * r1_loss * 12
                r1_loss_sum.backward()
                optimizer_D.step()

                data_X.requires_grad = False
                data_Y.requires_grad = False 
            
            
            #################################################################################################
            
            num_iter += 1
            # Output training stats
            if num_iter%5 == 0:
                print('it[{:04d}/{:04d}] \tLoss_D:{:.4f} \tLoss_G:{:.4f} \tLoss_I:{:.4f} \tLoss_C:{:.4f} \telapsed_time:{:.2f}mins'.format(
                    num_iter, max_iter, err_D.item(), err_G.item(), err_I.item(), err_C.item(), (time.time()-train_start_time)/60
                ))

            if num_iter%args.save_img==0 or num_iter==max_iter:
                save_name = osp.join(save_dir, 'it{:04d}.pt'.format(num_iter))
                save_name_w=osp.join(save_dir, 'final_weights.pt')

                #plt.imshow(outputs)
                
                if (num_iter%args.save_model==0 or num_iter==max_iter) and num_iter>10000:
                    torch.save({
                        'model_G_X': model_G_X_ema.state_dict(),
                        'model_G_Y': model_G_Y_ema.state_dict()
                    }, save_name)
            if num_iter%500==0:
                with torch.no_grad():
                    # model_G_X.eval()
                    # model_G_Y.eval()
                    for i, (input_batch, label_batch) in enumerate(test_mr_loader):
                        if i%100==0:
                            image_name = 'it{:04d}_{:04d}.jpg'.format(num_iter,i)
                            save_path = os.path.join(save_dir, image_name)
                            #print("3:",data_X.size())
                            data_X= input_batch.cuda().float()
                            #print("4:",data_X.size())
                            
                            data_Y, labels_Y = next(iter(test_ct_loader))
                            data_Y = data_Y.cuda().float()

                            output_X = model_G_X_ema(data_X)
                            output_Y = model_G_Y_ema(data_Y)
                            
                            cycle_X=model_G_Y_ema(output_X)
                            cycle_Y=model_G_X_ema(output_Y)
                            #print(output_X.size(),'X몇채널인지')
                            data_X=data_X.cpu().data.numpy()
                            data_X=(data_X+1)/2
                            data_Y=data_Y.cpu().data.numpy()
                            data_Y=(data_Y+1)/2
                            output_X=output_X.cpu().data.numpy()
                            output_X=(output_X+1)/2
                            output_Y=output_Y.cpu().data.numpy()
                            output_Y=(output_Y+1)/2
                            cycle_X=cycle_X.cpu().data.numpy()
                            cycle_X=(cycle_X+1)/2
                            cycle_Y=cycle_Y.cpu().data.numpy()
                            cycle_Y=(cycle_Y+1)/2
                            
                            output_X=output_X[0]
                            output_Y=output_Y[0]
                            data_X=data_X[0]
                            data_Y=data_Y[0]
                            cycle_X=cycle_X[0]
                            cycle_Y=cycle_Y[0]
                            # print(data_X.shape,'data_X.shape')
                            # print(data_Y.shape,'data_Y.shape')
                            # print(output_X.shape,'output_X.shape')
                            # print(output_Y.shape,'output_Y.shape')
                            
                            # output_X=output_X.squeeze()
                            # output_Y=output_Y.squeeze()
                            # data_X=data_X.squeeze()
                            # data_Y=data_Y.squeeze()
                            output_X1, output_X2, output_X3=output_X[0],output_X[1],output_X[2]
                            data_X1, data_X2, data_X3=data_X[0],data_X[1],data_X[2]
                            output_Y1, output_Y2, output_Y3=output_Y[0],output_Y[1],output_Y[2]
                            data_Y1, data_Y2, data_Y3=data_Y[0],data_Y[1],data_Y[2]
                            cycle_X1, cycle_X2, cycle_X3=cycle_X[0],cycle_X[1],cycle_X[2]
                            cycle_Y1, cycle_Y2, cycle_Y3=cycle_Y[0],cycle_Y[1],cycle_Y[2]
                            outputs_X = np.concatenate((output_X1, output_X2, output_X3), axis=1)
                            outputs_Y = np.concatenate((output_Y1, output_Y2, output_Y3), axis=1)
                            datas_X = np.concatenate((data_X1, data_X2, data_X3), axis=1)
                            datas_Y = np.concatenate((data_Y1, data_Y2, data_Y3), axis=1)
                            cycles_X=np.concatenate((cycle_X1, cycle_X2, cycle_X3), axis=1)
                            cycles_Y=np.concatenate((cycle_Y1, cycle_Y2, cycle_Y3), axis=1)
                            outputs = np.concatenate((datas_X, outputs_X, cycles_X, datas_Y, outputs_Y,cycles_Y), axis=0)
                            plt.imshow(outputs, cmap='gray')
                            plt.savefig(save_path, dpi=300)
                            plt.close()
                            plt.pause(0.001)
import random
from PIL import ImageFile
import numpy as np
import copy
import os
import torch
import torchvision.transforms as ttransforms
import medpy.io as medio
from Cityscapes import Cityscapes


ImageFile.LOAD_TRUNCATED_IMAGES = True

import SimpleITK as sitk
class FUDAN_25d(Cityscapes):
        
    def __init__(self,
                 list_path='/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/FUDAN_2d/',
                 contrast='mr',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3:3, 4:4}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        
        
        slice_id = image_path.split('_')[-1].split('.')[0]
        
        
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id)+1)  + '.nii.gz'

        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)

        if not os.path.exists(image_path_previous) :
            slice_id_new = str(int(slice_id)+1)
            
        elif not os.path.exists(image_path_after) :
            slice_id_new = str(int(slice_id)-1)
        
        else : 
            slice_id_new = slice_id
            
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id_new)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id_new)+1)  + '.nii.gz'
        slice_id_current   = '_' + str(int(slice_id_new))  + '.nii.gz'
        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)
        image_path = image_path.replace(ttt, slice_id_current)

        
        image_0 , _ = medio.load(image_path_previous)
        image_1 , _ = medio.load(image_path)
        image_2 , _ = medio.load(image_path_after)
        
        
        
        
        # print(image_0.shape, 'image-0_shape')
        # print(image_1.shape, 'image-1-shape')
        # print(image_2.shape, 'image-2_shape')
        
        image= np.concatenate([image_0, image_1, image_2], axis=-1).astype(np.float16)
        # print(image.shape, 'image_shape')

        image = (image -np.min(image))  / (np.max(image)-np.min(image) + 1e-9) *2 -1
        
        image=  torch.tensor(image)

        # print(image.shape, 'image_shape')
        # print(image.shape, 'iamge_shape')
        
        # if len(image.shape)==2 : 
        #     image=  image[:,:,None]
        
        
        # if self.split =='train' and self.contrast == 't1':
            
            
        gt_image_path = image_path.replace('image','label').replace('img','label')
        
        gt_image_path_previous = gt_image_path.replace(ttt, slice_id_previous)
        gt_image_path_after = gt_image_path.replace(ttt, slice_id_after)
        gt_image_path = gt_image_path.replace(ttt, slice_id_current)

        gt_image0, _ =medio.load(gt_image_path_previous)
        gt_image1, _ =medio.load(gt_image_path)
        gt_image2, _ =medio.load(gt_image_path_after) 
        

        gt_image= np.concatenate([gt_image0, gt_image1, gt_image2], axis=-1).astype(np.float16)
        gt_image=  torch.tensor(gt_image)
        # gt_image = torch.tensor(gt_image)[:,:,None]
        # gt_image = gt_image[:,:, z_choice-int((self.in_ch-1)/2) : z_choice+(int(self.in_ch/2)+1)]



        # else : 
        #     gt_image = torch.zeros_like(image)


        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
                
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
        # final transform
        if mask is not None:
            #print(mask.size())
            mask = mask[0]
            img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
            #print(mask.size())
            return img, mask
        else:
            img = self._img_transform_BraTS(img)
            return img





    def _val_sync_transform_BraTS(self, img, mask):
        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)

        
        #img=  torch.cat([img, img, img], dim=0)
        # final transform
        mask = mask[0]
        
        
        img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
        return img, mask

    def _img_transform_BraTS(self, image):
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float16)
        
        #target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    
class FUDAN_2d(Cityscapes):
        
    def __init__(self,
                 list_path='/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/FUDAN_2d/',
                 contrast='mr',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3:3, 4:4}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        
        
        slice_id = image_path.split('_')[-1].split('.')[0]
        
        
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id)+1)  + '.nii.gz'

        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)

        if not os.path.exists(image_path_previous) :
            slice_id_new = str(int(slice_id)+1)
            
        elif not os.path.exists(image_path_after) :
            slice_id_new = str(int(slice_id)-1)
        
        else : 
            slice_id_new = slice_id
            
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id_new)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id_new)+1)  + '.nii.gz'
        slice_id_current   = '_' + str(int(slice_id_new))  + '.nii.gz'
        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)
        image_path = image_path.replace(ttt, slice_id_current)

        
        image_0 , _ = medio.load(image_path_previous)
        image_1 , _ = medio.load(image_path)
        image_2 , _ = medio.load(image_path_after)
        
        
        
        
        # print(image_0.shape, 'image-0_shape')
        # print(image_1.shape, 'image-1-shape')
        # print(image_2.shape, 'image-2_shape')
        
        #image= np.concatenate([image_0, image_1, image_2], axis=-1).astype(np.float16)
        image=image_1
        # print(image.shape, 'image_shape')

        image = (image -np.min(image))  / (np.max(image)-np.min(image) + 1e-9) *2 -1
        
        image=  torch.tensor(image)

        #print(image.shape, 'image_shape')
        # print(image.shape, 'iamge_shape')
        
        # if len(image.shape)==2 : 
        #     image=  image[:,:,None]
        
        
        # if self.split =='train' and self.contrast == 't1':
            
            
        gt_image_path = image_path.replace('image','label').replace('img','label')
        
        gt_image_path_previous = gt_image_path.replace(ttt, slice_id_previous)
        gt_image_path_after = gt_image_path.replace(ttt, slice_id_after)
        gt_image_path = gt_image_path.replace(ttt, slice_id_current)

        gt_image0, _ =medio.load(gt_image_path_previous)
        gt_image1, _ =medio.load(gt_image_path)
        gt_image2, _ =medio.load(gt_image_path_after) 
        

        gt_image= torch.tensor(gt_image1)
        #print(gt_image.size(),'asdfasdf')
        # gt_image = torch.tensor(gt_image)[:,:,None]
        # gt_image = gt_image[:,:, z_choice-int((self.in_ch-1)/2) : z_choice+(int(self.in_ch/2)+1)]



        # else : 
        #     gt_image = torch.zeros_like(image)


        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
        #print(image.size(),'FFFFFFFFFFFFFFFF')        
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        #print(image.size(),'123123123')
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
        # final transform
        if mask is not None:
            #print(mask.size())
            mask = mask[0]
            img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
            #print(mask.size())
            return img, mask
        else:
            img = self._img_transform_BraTS(img)
            return img





    def _val_sync_transform_BraTS(self, img, mask):
        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)

        
        #img=  torch.cat([img, img, img], dim=0)
        # final transform
        # mask = mask[0]
        
        
        img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
        return img, mask

    def _img_transform_BraTS(self, image):
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float16)
        
        #target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    

def get_dataset_cardiac(imsize, batch_size=4):
    train_dataset_mr  = FUDAN_25d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='mr', split='train',crop_size=imsize, train=True, in_ch = 1)
    train_dataloader_mr = torch.utils.data.DataLoader(train_dataset_mr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_dataset_ct  = FUDAN_25d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='ct', split='train',crop_size=imsize, train=True, in_ch = 1)
    train_dataloader_ct = torch.utils.data.DataLoader(train_dataset_ct, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset_mr  = FUDAN_25d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='mr', split='test',crop_size=imsize, train=False, in_ch = 1)
    test_dataloader_mr = torch.utils.data.DataLoader(test_dataset_mr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset_ct  = FUDAN_25d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='ct', split='test',crop_size=imsize, train=False, in_ch = 1)
    test_dataloader_ct = torch.utils.data.DataLoader(test_dataset_ct, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataloader_mr,train_dataloader_ct,test_dataloader_mr,test_dataloader_ct

def get_dataset_cardiac_2d(imsize, batch_size=4):
    train_dataset_mr  = FUDAN_2d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='mr', split='train',crop_size=imsize, train=True, in_ch = 1)
    train_dataloader_mr = torch.utils.data.DataLoader(train_dataset_mr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_dataset_ct  = FUDAN_2d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='ct', split='train',crop_size=imsize, train=True, in_ch = 1)
    train_dataloader_ct = torch.utils.data.DataLoader(train_dataset_ct, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset_mr  = FUDAN_2d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='mr', split='test',crop_size=imsize, train=False, in_ch = 1)
    test_dataloader_mr = torch.utils.data.DataLoader(test_dataset_mr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset_ct  = FUDAN_2d(list_path='/home/compu/SJJ/DRANet-master/DRANet-master/data_list/FUDAN_2d', 
                                  contrast='ct', split='test',crop_size=imsize, train=False, in_ch = 1)
    test_dataloader_ct = torch.utils.data.DataLoader(test_dataset_ct, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataloader_mr,train_dataloader_ct,test_dataloader_mr,test_dataloader_ct

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, einsum
from functools import partial
from TimeSformer.timesformer.models.vit import VisionTransformer
from einops import rearrange
from math import sqrt


class SDC_UDA_Deep(nn.Module):
    
    def __init__(self, input_nc, output_nc):
        super(SDC_UDA_Deep,self).__init__()
        
        
        self.input_nc=input_nc
        self.output_nc=output_nc
        #(256,256)에서 
        self.enc1=nn.Sequential(nn.ReflectionPad2d(3), #(262,262)
                                nn.Conv2d(self.input_nc,8,7),#(256,256)
                                nn.InstanceNorm2d(8),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(8,8,3,1,1), #(256,256)
                                nn.InstanceNorm2d(8),
                                nn.ReLU(inplace=True)
                                )
        self.down1=nn.Sequential(nn.Conv2d(8,16,3,2,1), #(128,128)
                                 nn.InstanceNorm2d(16),
                                 nn.ReLU(inplace=True))
        
        self.enc2=nn.Sequential(nn.Conv2d(16,16,3,1,1), #(128,128)
                                nn.InstanceNorm2d(16),
                                nn.ReLU(inplace=True)
                                )
        #img_size=128
        self.timesformer1 = VisionTransformer(img_size=128, num_classes=3, patch_size=2, in_chans=16, embed_dim=96, depth=2, 
                                              num_heads=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=3)
        #나오면 96으로
        self.enc3=nn.Sequential(nn.Conv2d(96,96,3,1,1),
                                nn.InstanceNorm2d(96),
                                nn.ReLU(inplace=True)
                                )
        #embedding 96->384로 4배
        self.timesformer2 = VisionTransformer(img_size=64, num_classes=3, patch_size=2, in_chans=96, embed_dim=384, depth=4, 
                                              num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=3)
        
        #out_features2 = 384
        
        self.enc4=nn.Sequential(nn.Conv2d(384,384,3,1,1),
                                nn.InstanceNorm2d(384),
                                nn.ReLU(inplace=True)
                                )
        
        self.up1=nn.Sequential(nn.ConvTranspose2d(384,192,3,2,1,1),
                                     nn.InstanceNorm3d(192),
                                     nn.ReLU(inplace=True)
                                     )
        self.dec1=nn.Sequential(
                                     nn.Conv2d(192+96,96,3,1,1),
                                     nn.InstanceNorm2d(96),
                                     nn.ReLU(inplace=True)
                                )
        self.up2=nn.Sequential(nn.ConvTranspose2d(96,48,3,2,1,1),
                                     nn.InstanceNorm3d(48),
                                     nn.ReLU(inplace=True)
        )
        self.dec2=nn.Sequential(
                                     nn.Conv2d(48+16,16,3,1,1),
                                     nn.InstanceNorm2d(16),
                                     nn.ReLU(inplace=True)
                                )
        
        self.up3=nn.Sequential(nn.ConvTranspose2d(16,8,3,2,1,1),
                                     nn.InstanceNorm3d(8),
                                     nn.ReLU(inplace=True)
        )
        self.dec3=nn.Sequential(
                                     nn.Conv2d(16,16,3,1,1),
                                     nn.InstanceNorm2d(16),
                                     nn.ReLU(inplace=True)
                                )
        
        self.last_conv=nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(16,64,7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64,self.output_nc,1),
                                     nn.Tanh())
    
    def forward(self,x):
        B,T,H,W = x.shape
        #print('a',x.size())
        x= x.view(-1,1,H,W).contiguous() #(3B,1,256,256)
        #print('b',x.size())
        x=self.enc1(x)
        
        x1=self.down1(x)
        x1=self.enc2(x1)
        BT,C,H,W=x1.shape
        x1_=x1.reshape(B,C,3,H,W)
        #print('c',x1_.size())
        x2=self.timesformer1(x1_)
        #print('d',x2.size())
        x2=self.enc3(x2)
        #print(x2.size())
        BT,C,H,W=x2.shape
        x2_=x2.reshape(B,C,3,H,W)
        #print('x2 after reshape', x2_.size())
        x3=self.timesformer2(x2_)
        x3=self.enc4(x3)
        
        x3=self.up1(x3)
        x3=self.dec1(torch.cat([x3,x2], dim=1))
        x3=self.up2(x3)
        x3=self.dec2(torch.cat([x3,x1],dim=1))
        x3=self.up3(x3)
        x3=self.dec3(torch.cat([x3,x],dim=1))
        x=self.last_conv(x3)
        BC,A,H,W=x.shape
        x=x.reshape(B,T,H,W)
        #print(x.shape,'final shape')
        return x
    
    
class SDC_UDA(nn.Module):
    
    def __init__(self, input_nc, output_nc):
        super(SDC_UDA,self).__init__()
        
        
        self.input_nc=input_nc
        self.output_nc=output_nc
        #(256,256)에서 
        self.enc1=nn.Sequential(nn.ReflectionPad2d(3), #(262,262)
                                nn.Conv2d(self.input_nc,32,7),#(256,256)
                                nn.InstanceNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32,32,3,1,1), #(256,256)
                                nn.InstanceNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32,32,3,1,1), #(256,256)
                                nn.InstanceNorm2d(32),
                                nn.ReLU(inplace=True)
                                )
        self.down1=nn.Sequential(nn.Conv2d(32,64,3,2,1), #(128,128)
                                 nn.InstanceNorm2d(64),
                                 nn.ReLU(inplace=True))
        
        # self.enc2=nn.Sequential(nn.Conv2d(16,16,3,1,1), #(128,128)
        #                         nn.InstanceNorm2d(16),
        #                         nn.ReLU(inplace=True)
        #                         )
        #img_size=128
        self.timesformer1 = IISA(num_dims=(128,), num_heads=(8,), ff_expansion=(4,), reduction_ratio=(2,), num_layers=(4,), stage_kernel_stride_pad=((3,2,1),), feat_dim=64)
        #current_size : 128/8=16
        #나오면 96으로
        self.up1=nn.Sequential(nn.ConvTranspose2d(128,64,3,2,1,1), #64
                                nn.InstanceNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.up2=nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,1), #128
                                nn.InstanceNorm2d(32),
                                nn.ReLU(inplace=True),
                                )
        self.up3=nn.Sequential(nn.ConvTranspose2d(32,16,3,2,1,1), #256
                                nn.InstanceNorm2d(16),
                                nn.ReLU(inplace=True),                          
                                )
        #이거 끝나면 (512,512)
        self.up4=nn.Sequential( nn.ConvTranspose2d(16,8,3,2,1,1), #256
                                nn.InstanceNorm2d(8),
                                nn.ReLU(inplace=True),      
            
        )
        #embedding 96->384로 4배
        # self.timesformer2 = VisionTransformer(img_size=64, num_classes=3, patch_size=2, in_chans=96, embed_dim=384, depth=4, 
        #                                       num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        #                                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=3)
        
        #out_features2 = 384
        

        self.dec1=nn.Sequential(
                                     nn.Conv2d(32+32,64,3,1,1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64,64,3,1,1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                )
        
        
        self.last_conv=nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(64,64,7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64,self.output_nc,1),
                                     nn.Tanh())
    
    def forward(self,x):
        B,T,H,W = x.shape
        #print('a',x.size())
        x= x.view(-1,1,H,W).contiguous() #(3B,1,256,256)
        stack = []
        output = x
        output = self.enc1(output)
        stack.append(output)
        output = self.down1(output)
        #print('b',x.size())
        #output_mid = output.clone()
        #output = output.view(B, -1, T, H//2, W//2)
        output = self.timesformer1(output)
        #print(output.size())
        #output = output.reshape(B*T,4,4,-1).permute(0,3,1,2).contiguous()
        #print(output.size(),'output.size()')
        output_img = output
        #print(output_img.size(),'output_img.size()')
        output_img=self.up1(output_img)
        #print(output_img.size(),'output_img.size() up1 후에')
        #output_img=self.up2(output_img)
        #print(output_img.size(),'output_img.size() up2 후에')
        #output_img=self.up3(output_img)
        #print(output_img.size(),'output_img.size() up3 후에')
        d_sample_layer = stack.pop()
        output_img=self.up2(output_img)
        #print(output_img.size(),'output_img.size() up4 후에')
        #print(output_img.size(),'output_img.size() concat 전에')
        #print(d_sample_layer.size(),'d_sample_layer.size() concat 전에')

        output_img=torch.cat([output_img, d_sample_layer],dim=1)
        output_img=self.dec1(output_img)
        output_img=self.last_conv(output_img).view(B,T,H,W).contiguous()
        #print(output_img.shape, "마지막")
        x= x.view(-1,1,H,W).contiguous()
        return output_img


# https://github.com/lucidrains/segformer-pytorch/tree/main

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
        
        
        self.to_q_b   = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv_b  = nn.Conv2d(dim, dim * 2, 1, stride=1, bias=False)
        

    def forward(self, x):
        b, _, h, w = x.shape
        heads = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        
        
        # Inter slice
        q_img, k_img, v_img = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))
        
        sim_img = einsum('b i d, b j d -> b i j', q_img, k_img) * self.scale
        attn_img = sim_img.softmax(dim=-1)
        out_img = einsum('b i j, b j d -> b i d', attn_img, v_img)
        out_img = rearrange(out_img, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        
        
        # Intra slice
        
        q_slice, k_slice, v_slice = (self.to_q_b(x), *self.to_kv_b(x).chunk(2, dim=1))
        q_slice, k_slice, v_slice = map(lambda t: rearrange(t, 'b (h c) x y -> (x y h) b c', h=heads), (q_slice, k_slice, v_slice))
        
        sim_batch = einsum('i b d, i c d -> i b c', q_slice, k_slice) * self.scale
        attn_batch = sim_batch.softmax(dim=-1)
        out_batch = einsum('i b c, i c d -> i b d', attn_batch, v_slice)
        out_batch = rearrange(out_batch, '(x y h) b c -> b (h c) x y', h=heads, x=h, y=w)
        
        return self.to_out(out_img + out_batch)




class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)



class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        stage_kernel_stride_pad,
    ):
        super().__init__()
        
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)
            layers = nn.ModuleList([])
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            x = overlap_embed(x)
            
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


   
class IISA(nn.Module):
    def __init__(  self , num_dims, num_heads, ff_expansion, reduction_ratio, num_layers, stage_kernel_stride_pad, feat_dim ):
        super().__init__()
        size_red=1
        dims         = num_dims
        heads        = num_heads
        ff_expansion = ff_expansion
        reduction_ratio = reduction_ratio
        stage_kernel_stride_pad = stage_kernel_stride_pad
        num_layers = num_layers
        for k in stage_kernel_stride_pad:
            print(k)
            size_red = size_red * k[1]                
        channels = feat_dim
        
                
        
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = len(dims)), (dims, heads, ff_expansion, reduction_ratio, num_layers))        

        assert all([*map(lambda t: len(t) ==  len(dims), (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four dims-stage are allowed, all keyword arguments must be either a single value or a tuple of dim values'


        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers,
            stage_kernel_stride_pad = stage_kernel_stride_pad
        )



    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)
        Last_layer = layer_outputs[-1]
        return Last_layer
    
    

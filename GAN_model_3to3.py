import torch.nn as nn
import torch.nn.functional as F
class Generator_A2B(nn.Module):
    def __init__(self, n_channels):
        super(Generator_A2B, self).__init__()
        self.enc=nn.Sequential(nn.ReflectionPad2d(3),
                               nn.Conv2d(n_channels,64,7,1,0),
                               nn.InstanceNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64,128,3,2,1),
                               nn.InstanceNorm2d(128),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128,256,3,2,1),
                               nn.InstanceNorm2d(256),
                               nn.ReLU(inplace=True),)
        self.Resblock=nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(256,256,3,1,0),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(256,256,3,1,0),
                                    nn.InstanceNorm2d(256),)
        self.dec=nn.Sequential(nn.ConvTranspose2d(256,128,3,2,1, output_padding=1),
                               nn.InstanceNorm2d(125),
                               nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(128,64,3,2,1, output_padding=1),
                               nn.InstanceNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.ReflectionPad2d(3),
                               nn.Conv2d(64, n_channels, 7,1,0),
                               nn.Tanh(),)

    def forward(self, x):
        #x = x.unsqueeze(0)
        #print('1:',x.size())
        B,T,H,W = x.shape
        x=self.enc(x)
        #print('2:',x.size())
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.dec(x)
        BC,A,H,W=x.shape
        #x=x.reshape(3*B,1,H,W)
        #x=x.squeeze(0)
        #print(x.size())
        #print(x.size())
        return x

class Generator_B2A(nn.Module): #두 가지 CNN을 사용한다.
    def __init__(self, n_channels):
        super(Generator_B2A, self).__init__()
        self.enc=nn.Sequential(nn.ReflectionPad2d(3),
                               nn.Conv2d(n_channels,64,7,1,0),
                               nn.InstanceNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64,128,3,2,1),
                               nn.InstanceNorm2d(128),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128,256,3,2,1),
                               nn.InstanceNorm2d(256),
                               nn.ReLU(inplace=True),)
        self.Resblock=nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(256,256,3,1,0),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(256,256,3,1,0),
                                    nn.InstanceNorm2d(256),)
        self.dec=nn.Sequential(nn.ConvTranspose2d(256,128,3,2,1, output_padding=1),
                               nn.InstanceNorm2d(125),
                               nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(128,64,3,2,1, output_padding=1),
                               nn.InstanceNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.ReflectionPad2d(3),
                               nn.Conv2d(64, n_channels, 7,1,0),
                               nn.Tanh(),)

    def forward(self, x):
        #x = x.unsqueeze(0)
        #print('3:',x.size())
        B,T,H,W = x.shape
        x=self.enc(x)
        #print('4:',x.size())
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.Resblock(x)+x
        x=self.dec(x)
        BC,A,H,W=x.shape
        #x=x.reshape(3*B,1,H,W)
        #x=x.squeeze(0)
        #print(x.size())
        #print(x.size())
        return x
    
class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.dis_x=nn.Sequential(
            nn.Conv2d(n_channels,64,4,2,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128,4,2,1),
            nn.InstanceNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256,4,2,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,512,4,1,1),
            nn.InstanceNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,3,4,1,1),
            )
    def forward(self, x):
        #print(x.shape,'dis 1')
        B,T,H,W = x.shape
        #print(x.shape,'ddd')
        #x= x.view(-1,1,H,W).contiguous()
        #print(x.shape,'dis 1.1')
        x=self.dis_x(x)
        #print(x.shape,'dis 2')
        B,C,H,W=x.shape
        x=x.reshape(B*C,1,H,W)
        #print(x.shape,'dis 3')
        #print(x.shape, 'dis 4')
        #x=x.squeeze(0)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# class Discriminator_B(nn.Module):
#     def __init__(self, n_channels):
#         super(Discriminator_B, self).__init__()
#         self.dis_x=nn.Sequential(
#             nn.Conv2d(n_channels,64,4,2,1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64,128,4,2,1),
#             nn.InstanceNorm2d(128), 
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128,256,4,2,1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256,512,4,1,1),
#             nn.InstanceNorm2d(512), 
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512,1,4,1,1),
#             )
#     def forward(self, x):
#         #print(x.shape,'dis 1')
#         B,T,H,W = x.shape
#         x= x.view(-1,1,H,W).contiguous()
#        # print(x.shape,'dis 1.1')
#         x=self.dis_x(x)
#         #print(x.shape,'dis 2')
#         BC,A,H,W=x.shape
#         x=x.reshape(B,T,H,W)
#         #print(x.shape,'dis 3')
#         #print(x.shape, 'dis 4')
#         #x=x.squeeze(0)
#         return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


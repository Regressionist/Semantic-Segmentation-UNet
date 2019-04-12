import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=False, leaky:float=None):
        super().__init__()
        #nf = ifnone(nf, ni)
        self.conv = nn.Conv2d(ni, nf*(scale**2), 1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x
    
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.conv1a=nn.Sequential(nn.Conv2d(3,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.conv1b=nn.Sequential(nn.Conv2d(64,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2a=nn.Sequential(nn.Conv2d(64,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.conv2b=nn.Sequential(nn.Conv2d(128,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3a=nn.Sequential(nn.Conv2d(128,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.conv3b=nn.Sequential(nn.Conv2d(256,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4a=nn.Sequential(nn.Conv2d(256,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.conv4b=nn.Sequential(nn.Conv2d(512,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5a=nn.Sequential(nn.Conv2d(512,1024,3,1,0),nn.BatchNorm2d(1024),nn.ReLU())
        self.conv5b=nn.Sequential(nn.Conv2d(1024,1024,3,1,0),nn.BatchNorm2d(1024),nn.ReLU())
        
        self.up1=PixelShuffle_ICNR(ni=1024, nf=512, scale=2,leaky=False,blur=False)
        self.conv6a=nn.Sequential(nn.Conv2d(1024,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.conv6b=nn.Sequential(nn.Conv2d(512,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        
        self.up2=PixelShuffle_ICNR(ni=512, nf=256, scale=2,leaky=False,blur=False)
        self.conv7a=nn.Sequential(nn.Conv2d(512,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.conv7b=nn.Sequential(nn.Conv2d(256,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        
        self.up3=PixelShuffle_ICNR(ni=256, nf=128, scale=2,leaky=False,blur=False)
        self.conv8a=nn.Sequential(nn.Conv2d(256,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.conv8b=nn.Sequential(nn.Conv2d(128,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        
        self.up4=PixelShuffle_ICNR(ni=128, nf=64, scale=2,leaky=False,blur=False)
        self.conv9a=nn.Sequential(nn.Conv2d(128,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.conv9b=nn.Sequential(nn.Conv2d(64,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        
        self.conv9_final=nn.Conv2d(64,20,1)
        
    def forward(self,input_image):
        conv1=self.conv1b(self.refpad(self.conv1a(self.refpad(input_image))))
        o1=self.maxpool1(conv1)
        
        conv2=self.conv2b(self.refpad(self.conv2a(self.refpad(o1))))
        o2=self.maxpool2(conv2)
        
        conv3=self.conv3b(self.refpad(self.conv3a(self.refpad(o2))))
        o3=self.maxpool3(conv3)
        
        conv4=self.conv4b(self.refpad(self.conv4a(self.refpad(o3))))
        o4=self.maxpool4(conv4)
        
        conv5=self.up1(self.conv5b(self.refpad(self.conv5a(self.refpad(o4)))))
        
        conv6=self.up2(self.conv6b(self.refpad(self.conv6a(self.refpad(self.merge(conv5,conv4))))))
        conv7=self.up3(self.conv7b(self.refpad(self.conv7a(self.refpad(self.merge(conv6,conv3))))))
        conv8=self.up4(self.conv8b(self.refpad(self.conv8a(self.refpad(self.merge(conv7,conv2))))))
        conv9_temp=self.conv9b(self.refpad(self.conv9a(self.refpad(self.merge(conv8,conv1)))))
        conv9=self.conv9_final(self.refpad(conv9_temp))
        if self.training:
            conv9_a=self.conv9_final(self.refpad(F.dropout(conv9_temp,p=0.3)))
            conv9_b=self.conv9_final(self.refpad(F.dropout(conv9_temp,p=0.3)))
            conv9_c=self.conv9_final(self.refpad(F.dropout(conv9_temp,p=0.3)))
            return conv9,conv9_a,conv9_b,conv9_c
        else:
            return conv9
    def refpad(self, x):
        return F.pad(x,(1,1,1,1),'reflect')   
    def merge(self,outputs,inputs):
        offset = outputs.size()[2] - inputs.size()[2]
        
        
        if offset%2!=0:
            padding=2*[offset//2,(offset//2)+1]
        else:
            padding = 2 * [offset // 2, offset // 2]    
        
        output = F.pad(inputs, padding)
        
        return torch.cat([output,outputs],1)
    

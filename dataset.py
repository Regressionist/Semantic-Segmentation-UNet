import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import ImageFile
import skimage
from skimage import io, transform
import matplotlib.pyplot as plt
import scipy.misc as m
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import torch


class PlacesDataset(data.Dataset):
    def __init__(self,dic,augment,transforms=False):
        self.dic=dic
        self.transforms=transforms
        self.augments=augment
        self.img_size=(512, 1024)
        self.colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

        self.label_colours = dict(zip(range(19), self.colors))
        self.n_classes = 19
        
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        length=len(self.dic['image'])
        return length
    def __getitem__(self,idx):
        img_path=self.dic['image'][idx]
        mask_path=self.dic['mask'][idx]
        
        img = io.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = io.imread(mask_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        if self.augments==0:
            x=np.random.randint(low=0,high=512)
            y=np.random.randint(low=0,high=1024)
            img=img[x:x+512,y:y+1024,:]
            lbl=lbl[x:x+512,y:y+1024]
        #if self.augments==1:
         #   img=img[:512,:1024,:]
         #   lbl=lbl[:512,:1024]    
        #elif self.augments==2:
         #   img=img[512:,1024:,:]
         #   lbl=lbl[512:,1024:]
        #elif self.augments==3:
         #   img=img[:512,1024:,:]
          #  lbl=lbl[:512,1024:]
        #elif self.augments==4:
         #   img=img[512:,:1024,:]
         #   lbl=lbl[512:,:1024]
        #elif self.augments==5:
         #   img=img[256:768,512:1536,:]
         #   lbl=img[256:768,512:1536]
            
        if self.transforms==True:
            img,lbl=self.transform(img,lbl)
        
        return img, lbl
            
    def encode_segmap(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def decode_segmap(self,temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
    
    def transform(self,img,lbl):
        if (img.shape!=(512,1024,3)):
            img = m.imresize(img, (self.img_size[0], self.img_size[1],3))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        #img -= self.mean
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)
        classes = np.unique(lbl)
        if lbl.shape!=(512,1024):
            lbl = lbl.astype(float)
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
import numpy as np
import skimage
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from dataset import PlacesDataset
from data_dicts_hpc import Dicts
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam,SGD
from sklearn.metrics import jaccard_similarity_score as jss
from skimage.segmentation import find_boundaries as fb
from unet_model import Unet
from loss_functions import train_loss_function_variance, val_loss_function
import os

def safe_mkdir(path):
    "Creates a directory if there isn't one already."
    try:
        os.mkdir(path)
    except OSError:
        pass

def create_model_save_folders():
    safe_mkdir('/data/sk7685/cityscapes')
    safe_mkdir('/data/sk7685/cityscapes/checkpoints')

def get_loaders(aug=0):
    dic=Dicts()
    train_files,val_files=dic.get_dicts()

    train_dataset=PlacesDataset(train_files,augment=aug, transforms=True)
    val_dataset=PlacesDataset(val_files,augment=10,transforms=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=2)
    validation_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1)

    return train_loader, validation_loader



def train(unet,input_image,target_image,masks,optimizer):
    unet.train()
    optimizer.zero_grad()
    output_image,output_image_b,output_image_c,output_image_d=unet(input_image)
    loss=train_loss_function_variance(output_image,output_image_b,output_image_c,output_image_d,target_image,masks,unet.lambda_varloss)
    loss.backward()
    optimizer.step()
    return loss.item()

def validation(unet,input_image,target_image):
    with torch.no_grad():
        unet.eval()
        output_image=unet(input_image)
        loss=val_loss_function(output_image,target_image)
        return loss

def train_and_val(unet, start_point=0,end_point=10, aug=0, version = 'v0',weights_version_load='v0',weights_version_save='v0',optimizer='adam',lr=1e-3,drop_p=.2,lambda_varloss=100):
    train_loader, validation_loader = get_loaders(aug)
    create_model_save_folders()
    log_path = f'log_{version}.txt'
    checkpoint_path = '/data/sk7685/cityscapes/checkpoints'
    if(not os.path.isfile(log_path)):
        with open(log_path ,'a') as lgfile:
            lgfile.write(f'version:\t{version}\taugment\t{aug}\toptimizer\t{optimizer}\tlr\t{lr}\tdrop_p\t{drop_p}\tlambda\t{lambda_varloss}\tw_vers_load\t{weights_version_load}\tw_vers_save\t{weights_version_save}\n\n') 
            lgfile.write('step\tbatch\tt_loss\tval_IOU\n') 
            lgfile.flush()
    else:
        with open(log_path ,'a') as lgfile:
            lgfile.write(f'version:\t{version}\taugment\t{aug}\toptimizer\t{optimizer}\tlr\t{lr}\tdrop_p\t{drop_p}\tlambda\t{lambda_varloss}\tw_vers_load\t{weights_version_load}\tw_vers_save\t{weights_version_save}\n\n') 
            lgfile.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if optimizer == 'adam':
        optimizer = Adam(unet.parameters())
    elif optimizer == 'sgd':
        optimizer = SGD(unet.parameters(),lr=lr)

    checkpoint_path_load = f'{checkpoint_path}/model_{weights_version_load}.pth'
    if(os.path.isfile(checkpoint_path_load)):
        checkpoint = torch.load(checkpoint_path_load)
        unet.load_state_dict(checkpoint['model_state_dict'])
        batch = checkpoint['epoch']
        tl = checkpoint['tl']
        vl = checkpoint['vl']
        unet.train()

    tl=[]
    vl=[]
    benchmark=0
    for batch in range(start_point,end_point):
        running_loss=0
        running_loss_count=0
        for i,inputs in enumerate(train_loader):
            input_image=Variable(inputs[0]).to(device)
            target_image=Variable(inputs[1]).to(device)
            masks=Variable(inputs[2]).to(device)
            loss=train(unet,input_image,target_image,masks,optimizer)
            running_loss+=loss
            running_loss_count+=1
            tl.append(loss)
            if (i+1)%93==0:
                loss_val=0
                for j,inputs in enumerate(validation_loader):
                    input_image=(inputs[0]).to(device)
                    target_image=(inputs[1]).to(device)
                    loss_val+=validation(unet,input_image,target_image)
                loss_val=loss_val/(j+1)
                vl.append(loss_val)
                with open(log_path ,'a') as lgfile:
                    lgfile.write('{}\t{}\t{:.4}\t{:.4}\n'.format(int((i+1)/93), batch+1, running_loss/running_loss_count, loss_val))
                    lgfile.flush()
                running_loss=0
                running_loss_count=0
                if(loss_val>benchmark):
                    print('%--Saving the model--%')
                    torch.save({
                        'step':int((i+1)/93),
                        'epoch': batch+1,
                        'model_state_dict': unet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'tl': tl,
                        'vl': vl
                        }, f'{checkpoint_path}/model_{weights_version_save}.pth')
                    benchmark=loss_val
                
                

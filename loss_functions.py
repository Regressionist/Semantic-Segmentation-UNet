import torch
from sklearn.metrics import jaccard_similarity_score as jss
from skimage.segmentation import find_boundaries as fb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train_loss_function(input_image,target_image,masks):
    n,c,h,w=input_image.size()
    nt,ht,wt=target_image.size()
    assert c==20
    if (h!=ht and w!=wt):
        input_image=F.interpolate(input_image, size=(ht, wt), mode="bilinear", align_corners=True)
        
    input_image=input_image.transpose(1,2).transpose(2,3).contiguous()
    target_image[target_image==255]=19
     
    for i in range(target_image.size(0)):
        mask=masks[i]
        if i==0:
            loss=(F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')*mask.view(-1)).mean()
        else:
            loss+=(F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')*mask.view(-1)).mean()
        
    return loss

def train_loss_function_variance(input_image,input_image_b,input_image_c,input_image_d,target_image,masks,lambda_varloss):
    n,c,h,w=input_image.size()
    nt,ht,wt=target_image.size()
    assert c==20
    if (h!=ht and w!=wt):
        input_image=F.interpolate(input_image, size=(ht, wt), mode="bilinear", align_corners=True)
        input_image_b=F.interpolate(input_image_b, size=(ht, wt), mode="bilinear", align_corners=True)
        input_image_c=F.interpolate(input_image_c, size=(ht, wt), mode="bilinear", align_corners=True)
        input_image_d=F.interpolate(input_image_d, size=(ht, wt), mode="bilinear", align_corners=True)
        
        
    output_cat=torch.cat((input_image_b.unsqueeze(-1),input_image_c.unsqueeze(-1),input_image_d.unsqueeze(-1)),dim=-1)
    mean_var_loss=(torch.mean(torch.var(output_cat,dim=-1),dim=1)*(1-F.softmax(input_image,dim=1)[:,-1,:,:])).mean()
    input_image=input_image.transpose(1,2).transpose(2,3).contiguous()
    target_image[target_image==255]=19
     
    for i in range(target_image.size(0)):
        mask=masks[i]
        if i==0:
            loss=(F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')*mask.view(-1)).mean()
        else:
            loss+=(F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')*mask.view(-1)).mean()
        
    return loss+lambda_varloss*mean_var_loss

def val_loss_function(output_model,target):
    n,c,h,w=output_model.size()
    nt,ht,wt=target.size()
    
    if (h!=ht and w!=wt):
        output_model=F.interpolate(output_model, size=(ht, wt), mode="bilinear", align_corners=True)
        
    output_model=F.softmax(output_model.squeeze(0),dim=0).transpose(0,1).transpose(1,2)
    predicted=np.zeros((ht,wt))
    target=target.squeeze(0).cpu().numpy().reshape(-1)
    output_model=output_model.cpu().numpy()
    predicted=np.argmax(output_model,axis=2)
    predicted=predicted.reshape(-1)
    predicted[predicted==19]=255
    score=jss(target,predicted)
    return score

def FocalLoss(input_image, target, masks,gamma=2):
    n,c,h,w=input_image.size()
    nt,ht,wt=target_image.size()
    
    if (h!=ht and w!=wt):
        input_image=F.interpolate(input_image, size=(ht, wt), mode="bilinear", align_corners=True)
    
    input_image=input_image.transpose(1,2).transpose(2,3).contiguous()
    target_image[target_image==255]=19
    for i in range(target_image.size(0)):
        mask=masks[i]
        #eps=10**(-12)
        #counts=torch.bincount(target_image[i].view(-1))
        #counts=torch.sum(counts)/(counts.float()+eps)
        #print(counts.size())
        
        #alpha=torch.tensor([counts[j] for j in target_image[i].view(-1)]).type(torch.cuda.FloatTensor)
        #print(alpha.size())
        
        if i==0:
            logpt=F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')
            pt=torch.exp(-logpt)
            loss=((((1-pt)**gamma)*logpt*mask.view(-1))).mean()
        else:
            logpt=F.cross_entropy(input_image[i].view(-1,c),target_image[i].view(-1),reduction='none')
            pt=torch.exp(-logpt)
            loss+=((((1-pt)**gamma)* logpt*mask.view(-1))).mean()
    
    return loss

import os
from os import listdir
from os.path import isfile, join

class Dicts():
    def __init__(self):
        self.train_path='/data/sk7685/cityscapes/data/data_img/train/'
        self.val_path='/data/sk7685/cityscapes/data/data_img/val/'
        self.test_path='/data/sk7685/cityscapes/data/data_img/test/'
        self.mask_path_train='/data/sk7685/cityscapes/data/data_mask/train/'
        self.mask_path_val='/data/sk7685/cityscapes/data/data_mask/val/'
        self.mask_path_test='/data/sk7685/cityscapes/data/data_mask/test/'
        
    def get_dicts(self):
        train_files={}
        train_files['image']=[]
        train_files['mask']=[]
        
        test_files={}
        test_files['image']=[]
        test_files['mask']=[]
        
        val_files={}
        val_files['image']=[]
        val_files['mask']=[]
        
        train_folders = [f for f in listdir(self.train_path) if os.path.isdir(join(self.train_path, f))]
        val_folders = [f for f in listdir(self.val_path) if os.path.isdir(join(self.val_path, f))]
        test_folders = [f for f in listdir(self.test_path) if os.path.isdir(join(self.test_path, f))]
        
        for folder in train_folders:
            files=[f for f in listdir(self.train_path+folder) if isfile(join(self.train_path+folder, f))]
            for f in files:
                train_files['image'].append(self.train_path+folder+'/'+f)
                if isfile(join(self.mask_path_train+'/'+folder,'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')):
                    train_files['mask'].append(self.mask_path_train+folder+'/'+'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')
                    
        for folder in val_folders:
            files=[f for f in listdir(self.val_path+folder) if isfile(join(self.val_path+folder, f))]
            for f in files:
                val_files['image'].append(self.val_path+folder+'/'+f)
                if isfile(join(self.mask_path_val+'/'+folder,'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')):
                    val_files['mask'].append(self.mask_path_val+folder+'/'+'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')
                    
        for folder in test_folders:
            files=[f for f in listdir(self.test_path+folder) if isfile(join(self.test_path+folder, f))]
            for f in files:
                test_files['image'].append(self.test_path+folder+'/'+f)
                if isfile(join(self.mask_path_test+'/'+folder,'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')):
                    test_files['mask'].append(self.mask_path_test+folder+'/'+'_'.join(f.split('_',-1)[0:3])+'_gtFine_labelIds.png')
                    
                    
        return train_files,val_files
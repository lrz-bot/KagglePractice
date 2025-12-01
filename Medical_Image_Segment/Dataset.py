import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

data_path=r'E:\Kaggle\src\Medical Image Segment\Data\archive\lgg-mri-segmentation\kaggle_3m'
data_path=Path(data_path)
dataset_split_rate=3/7      
seed=42

def collect_pairs(data_path):
    image_path_list=[]
    mask_path_list=[]
    for img in data_path.rglob("*.tif"):
        p=img.stem()
        if p.endwith("_mask"):
            continue

        mask_path_stem=f"{p}_mask"
        mask_path=img.with_stem(mask_path_stem)
        if mask_path.exists():
            image_path_list.append(str(img))
            mask_path_list.append(str(mask_path))
    print("Total pairs found:", len(image_path_list))
    
    return image_path_list,mask_path_list

def split_data(image_path_list,mask_path_list,random_seed,split_rate):
    X_path_train,X_path_test,y_path_train,y_path_test=train_test_split(
        image_path_list,
        mask_path_list,
        test_size=split_rate,
        shuffle=True,
        random_state=random_seed
    )
    train_path_pairs=list(zip(X_path_train,y_path_train))
    test_path_pairs=list(zip(X_path_test,y_path_test))
    print(f"Train Pairs :{train_path_pairs.len()}")
    print(f"Test Pairs {test_path_pairs.len()}")
    
    return train_path_pairs,test_path_pairs

class FaceDataset(Dataset):
    def __init__(self,config):
        self.X=None
        self.y=None
        self.is_test=config['is_test']
        self.file_name=config["file_name"]
        self.sigma=config['sigma']
        self.heatmap_stride=config['stride']
        self.is_dropna=config['is_dataset_dropna']
    def load(self):
        df=pd.read_csv(self.file_name)
        sample_num=df.shape[0]
        df['Image']=df['Image'].apply(lambda im : np.fromstring(im,sep=' '))
        if self.is_dropna:
            df = df.dropna()   
            
        X=df.iloc[:,-1].apply(lambda img:img.reshape(96,96))
        X=X.to_numpy()
        
        
        if not self.is_test:
            y=(df.iloc[:,:-1]).to_numpy()
            y_shape_first=y.shape[0]
            y=y.reshape(y_shape_first,-1,2)
        self.X=X
        self.y=y
        
        print(f"原始样本总数：{sample_num}, 加载样本数：{X.shape[0]}")
        return X,y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        H,W=96,96
        X=self.X[index]
        X=X/255.
        X=(X[np.newaxis,:,:])
        X = torch.from_numpy(X).float()    #添加通道维度
        y_heatmap=(self._put_Gaussian_heatmaps((self.y[index]),H,W,self.heatmap_stride,self.sigma))
        y_heatmap=torch.from_numpy(y_heatmap).float()
        y_orig=((self.y[index]))
        y_orig=torch.from_numpy(y_orig).float()
        if(self.is_test):
            return X
        return X,y_heatmap,y_orig
    
    def _put_Gaussian_heatmap(self,crop_x_size,crop_y_size,center,stride,sigma):
        sigma=self.sigma
        grid_x=crop_x_size/stride
        grid_y=crop_y_size/stride  #热力图的网格大小
        x_range=np.array([i for i in range(int(grid_x))])
        y_range=np.array([i for i in range(int(grid_y))])
        xx,yy=np.meshgrid(x_range,y_range)
        xx_orgin=xx*stride+stride/2-0.5
        yy_orgin=yy*stride+stride/2-0.5  #将heatmap上点坐标换算为原图上坐标
        distance_square=(xx_orgin-center[0])**2+(yy_orgin-center[1])**2
        Gaussian_map=np.exp(-(distance_square/(sigma**2)))
        return Gaussian_map
    
    def _put_Gaussian_heatmaps(self,keypoints,crop_x_size,crop_y_size,stride,sigma):
        """
        keypoints:(15,2)
        """
        
        heatmaps_list=[]
        for i in range((keypoints.shape[0])):
            center=keypoints[i,:]
            if np.isnan(center).any():   #检测是否与任何缺失元素
                heatmaps_list.append(np.zeros((int(crop_x_size/stride),int(crop_y_size/stride))))
            else:
                heatmaps_list.append(self._put_Gaussian_heatmap(crop_x_size,crop_y_size,center,stride,sigma))
        heatmaps=np.stack(heatmaps_list,axis=0)
        return heatmaps
    


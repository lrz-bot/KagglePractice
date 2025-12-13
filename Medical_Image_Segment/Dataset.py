import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

data_path=r'E:\Kaggle\src\Medical Image Segment\Data\archive\lgg-mri-segmentation\kaggle_3m'
data_path=Path(data_path)
      


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
        print(f"Train Pairs :{len(train_path_pairs)}")
        print(f"Test Pairs {len(test_path_pairs)}")
        
        return train_path_pairs,test_path_pairs
def collect_pairs(data_path):
        image_path_list=[]
        mask_path_list=[]
        data_path=Path(data_path)
        for img in data_path.rglob("*.tif"):
            p=img.stem
            if p.endswith("_mask"):
                continue

            mask_path_stem=f"{p}_mask"
            mask_path=img.with_stem(mask_path_stem)
            if mask_path.exists():
                image_path_list.append(str(img))
                mask_path_list.append(str(mask_path))
        print("Total pairs found:", len(image_path_list))
        
        return image_path_list,mask_path_list

class MedicalDataset(Dataset):
    
    def __init__(self,config,path_pair):
        super().__init__()
        self.is_test=config['is_test']
        self.path_pair=path_pair
         
            
    def __len__(self):
         return len(self.path_pair)
    def __getitem__(self, index):
        img_path,mask_path=self.path_pair[index]
        img=Image.open(img_path)
        img=np.array(img, dtype=np.float32) / 255.0  # 转换为 float32 并归一化
        img=(img-0.5)*2
        
        mask=Image.open(mask_path)
        mask=np.array(mask, dtype=np.float32) / 255.0  # 处理 mask
             
             # 调整维度顺序：从 (H, W, C) 转换为 (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        if len(mask.shape) == 3:
                mask = np.transpose(mask, (2, 0, 1))
        else:
                mask = np.expand_dims(mask, 0)  # 如果是单通道，添加通道维度
             
        return torch.from_numpy(img), torch.from_numpy(mask)
        
        # 测试模式下也需要调整维度
        
        
    
    


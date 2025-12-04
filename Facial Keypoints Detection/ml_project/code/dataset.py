import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class FaceDataset(Dataset):
    def __init__(self,config):
        self.X=None
        self.y=None
        self.is_test=config['is_test']
        self.file_name=config["file_name"]
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
            self.y=y
            self.X=X
        
        
            print(f"原始样本总数：{sample_num}, 加载样本数：{X.shape[0]}")
            return X,y
        self.X=X
        print(f"原始样本总数：{sample_num}, 加载样本数：{X.shape[0]}")
        return X
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        H,W=96,96
        X=self.X[index]
        X=X/255.
        X=(X[np.newaxis,:,:])     #添加通道维度
        if not self.is_test:

            y=self.y[index,:]
            y=self.process_NaN(y)
            y=y/96    #归一化
            return torch.from_numpy(X).float(),torch.from_numpy(y).float()
        return torch.from_numpy(X).float()
    
    def process_NaN(self,y):
        """
        :param y: (15,2)
        """
        
        y_bool=np.isnan(y)
        y=np.nan_to_num(y)
        y_bool=(y_bool[:,::-1]+y_bool)[:,0]   # 使任何有缺失值的坐标的x ，y值都为 True
        if(y_bool[0]):
            for i in range(1,y_bool.shape[0]):
                if not y_bool[i]:
                    y[0,:]=y[i,:]
                    break

        for i in range(1,y_bool.shape[0]):
            if(y_bool[i]):
                for j in range(i-1,-1,-1):
                    if not (y_bool[j]):
                        y[i,:]=y[j,:]
                        break
        
        return y                
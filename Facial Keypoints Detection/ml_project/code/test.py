import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import facemodel
from dataset import FaceDataset
from torch.backends import cudnn
import torch.optim as optim
from evaluation import plot_sample
import matplotlib.pyplot as plt
from pathlib import Path
import os
config=dict()
config['is_test']=True
config['file_name']=r"E:\Kaggle\src\Facial Keypoints Detection\Data\test\test.csv"
config['is_batch_test']=False
config['checkout']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\weights\kd_epoch_1640_model.ckpt'
config['device']='cuda'
config['is_dataset_dropna']=False
config['img_save_path']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\test_img_res'
config['save_freq']=100
config['test_img_index']=20


if __name__=="__main__":
    print('------Getting Start to Test------')
    device=config['device']
    net=facemodel()
    net.float().to(device)
    net.eval()
    faceDataset=FaceDataset(config)
    img_np=faceDataset.load()
    
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout'],weights_only=True)) #防止恶意攻击
        if config['is_batch_test']:

            dataloader=DataLoader(dataset=faceDataset,batch_size=1,shuffle=False,num_workers=4)

            for i,X in enumerate(dataloader):
            
            
            
                X=X.to(device)
                output=net(X)
                
                if (i+1)%(config['save_freq'])==0:
                     
                    img_save_path=Path(config['img_save_path'])
                    diretory_path=img_save_path
                    
                    output=plot_sample(np.squeeze((X[0]).cpu().detach().numpy()),(output[0].cpu().detach().numpy()))
                    
                    output_img_sve_path=diretory_path  / f"index{i}_ouput_test_img.png"
                    plt.savefig(output_img_sve_path)        
                    plt.close(output)
        else:
                print(f"--- 测试图片形状为{img_np[config['test_img_index']].shape} ---")
                img_test_np=img_np[config['test_img_index']]
                img_test_np=img_test_np[np.newaxis,np.newaxis,:,:]
                img_test_tensor=torch.from_numpy(img_test_np).float()
                img_test_tensor=img_test_tensor/255.

                img_test_tensor=img_test_tensor.to(device)
                with torch.no_grad(): 
                    output = net(img_test_tensor)
                
                output_img=plot_sample(np.squeeze((img_test_tensor[0]).cpu().detach().numpy()),(output[0].cpu().detach().numpy()))
                plt.show()

        
                


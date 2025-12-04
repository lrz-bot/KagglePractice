import pandas as pd
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
config['checkout']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\weights\kd_epoch_1640_model.ckpt'
config['device']='cuda'
config['is_dataset_dropna']=False
config['submit_file_path']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\SampleSubmission.csv'
config['submit_lookup_file']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\IdLookupTable.csv'
label_dict=dict()

label_dict={
     'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29,
}

if __name__=="__main__":
    print('------Getting Start to Submit ------')
    device=config['device']
    net=facemodel()
    net.float().to(device)
    net.eval()
    faceDataset=FaceDataset(config)
    img_np=faceDataset.load()
    
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout'],weights_only=True)) #防止恶意攻击
        

        df_idlookup=pd.read_csv(config['submit_lookup_file'])
        df_submit=pd.read_csv(config['submit_file_path'])
        num_submit=len(df_submit)
        for i in range(num_submit):
            
            img_test_np=img_np[df_idlookup.iloc[i,1]-1]
            img_test_np=img_test_np[np.newaxis,np.newaxis,:,:]
            img_test_tensor=torch.from_numpy(img_test_np).float()
            img_test_tensor=img_test_tensor/255.

            img_test_tensor=img_test_tensor.to(device)
            with torch.no_grad(): 
                    output = net(img_test_tensor)
            output=output.cpu().detach().numpy()

            feature_name=df_idlookup.iloc[i,2]
            output=np.squeeze(output)
            output_number=output[label_dict[feature_name]]
            df_submit.iloc[i,1]=output_number
            if (i+1)%10==0:
                 
                print(f"--- 成功写入第{i+1}个数据 ---")

        
        df_submit.to_csv(config['submit_file_path'], index=False, encoding='utf-8-sig')
        print(f"\n数据已成功更新并覆盖写入文件: {config['submit_file_path']}")
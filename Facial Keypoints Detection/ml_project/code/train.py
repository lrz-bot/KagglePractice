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
config['lr']=0.01
config['is_test']=False
config['file_name']=r"E:\Kaggle\src\Facial Keypoints Detection\Data\training\training.csv"
config['batch_size']=64
config['shuffle']=True
config['checkout']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\weights\kd_epoch_10_model.ckpt'
config['start_epoch']=10
config['epoch_num']=200
config['device']='cuda'
config['save_freq']=10
config['pring_freq_batch']=10
config['momentum']=0.9
config['is_dataset_dropna']=False
config['img_save_path']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\train_res_img'
config['weights_save_path']=r'E:\Kaggle\src\Facial Keypoints Detection\Data\ml_project\weights'



if __name__=="__main__":
    print('------Getting Start to Train------')
    device=config['device']
    cudnn.benchmark=True    #**牺牲少量启动时间换取显著训练速度**的常见优化手段，适用于输入尺寸不变的模型。
    torch.manual_seed(0)
    net=facemodel()
    net.float().to(device)
    net.train()
    criterion=nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    faceDataset=FaceDataset(config)
    faceDataset.load()
    dataloader=DataLoader(dataset=faceDataset,batch_size=config['batch_size'],shuffle=config['shuffle'],num_workers=4)
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout'],weights_only=True)) #防止恶意攻击
    for epo in range(config['start_epoch'],config['start_epoch']+config['epoch_num']+1):
        running_loss=0.0
        for i,(X,y) in enumerate(dataloader):
            
            optimizer.zero_grad()    #   确保初始梯度为0，不会积累
            
            
            X=X.to(device)
            y=y.to(device)
            y = y.reshape(y.size(0), -1)
            output=net(X)
            output=output/96
            loss = criterion(output, y)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.5)   #梯度裁剪
            optimizer.step()
            running_loss+=loss.item()
            if (i + 1) % config['pring_freq_batch'] == 0:
                avg_loss = running_loss / (i+1)

                print(
                    '[ Epoch {:>3d} | Batch {:>5d} / {:>5d} ] Average Loss: {:.6f} '.format(
                        epo, i + 1, len(dataloader), avg_loss)
                    )
        
        
                

        if (epo) % config['save_freq'] == 0 or epo == config['epoch_num']:
            img_save_path=Path(config['img_save_path'])
            diretory_path=img_save_path
            orig_img=plot_sample(np.squeeze((X[0]).cpu().detach().numpy()),(y[0].cpu().detach().numpy())*96)
            orig_img_save_path=diretory_path / f"epoch{epo}_orig_img.png"
            
            plt.savefig(orig_img_save_path)
            plt.close(orig_img)
            output=plot_sample(np.squeeze((X[0]).cpu().detach().numpy()),(output[0].cpu().detach().numpy())*96)
            
            output_img_sve_path=diretory_path  / f"epoch{epo}_ouput_img.png"
            plt.savefig(output_img_sve_path)        
            plt.close(output)
            
            
            
            
            save_path=config['weights_save_path']
            save_path=Path(save_path)
            save_path=save_path / f"kd_epoch_{epo}_model.ckpt"
            torch.save(net.state_dict(),save_path)
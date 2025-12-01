import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model_face import HourglassSingle
from dataset import FaceDataset
from torch.backends import cudnn
import torch.optim as optim
from evalution import plot_sample, mse_eval,get_peak
import matplotlib.pyplot as plt
from pathlib import Path
import os
config=dict()
config['lr']=0.000001
config['is_test']=False
config['file_name']=r"E:\Kaggle\Data\Facial Keypoints Detection\Data\training\training.csv"
config['sigma']=5.
config['stride']=1     #热力图步长
config['batch_size']=72
config['shuffle']=False
config['checkout']=''
config['start_epoch']=1
config['epoch_num']=200
config['device']='cuda'
config['save_freq']=10
config['pring_freq']=10
config['momentum']=0.9
config['is_dataset_dropna']=False



if __name__=="__main__":
    print('------Getting Start to Train------')
    device=config['device']
    cudnn.benchmark=True    #**牺牲少量启动时间换取显著训练速度**的常见优化手段，适用于输入尺寸不变的模型。
    torch.manual_seed(0)
    net=HourglassSingle()
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
        
        for i,(X,heat_map,y) in enumerate(dataloader):
            running_loss=0.0
            optimizer.zero_grad()    #   确保初始梯度为0，不会积累
            
            
            # N, C, H, W = heat_map.shape
            X=X.to(device)
            heat_map=heat_map.to(device)
            # channel_has_data = heat_map.view(N, C, -1).sum(dim=2) != 0  #形状为(N,C)
            # mask = channel_has_data.view(N, C, 1, 1).float() # 形状 (N, C, 1, 1)
            output=net(X)
            # output=output*mask
            loss = criterion(output, heat_map)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.5)   #梯度裁剪
            optimizer.step()
            running_loss+=loss.item()
            if (i + 1) % config['pring_freq'] == 0:
                avg_loss = running_loss / config['pring_freq']
                mse_loss=mse_eval(get_peak(output.cpu().detach().numpy(),config['stride']),y.cpu().detach().numpy())

                print(
                    '[ Epoch {:>3d} | Batch {:>5d} / {:>5d} ] Average Loss: {:.6f}  MSE_eval: {:.3f}'.format(
                        epo, i + 1, len(dataloader), avg_loss,mse_loss)
                    )
        script_path = Path(os.path.abspath(__file__))
        diretory_path=script_path.parent
        orig_img=plot_sample(np.squeeze((X[0]).cpu().detach().numpy()),y[0,:,:])
        orig_img_save_path=diretory_path / "train_eval_img" / f"epoch{epo}_orig_img.png"
        
        plt.savefig(orig_img_save_path)
        plt.close(orig_img)
        output=plot_sample(np.squeeze((X[0]).cpu().detach().numpy()),(get_peak(output.cpu().detach().numpy(),config['stride']))[0,:,:])
        
        output_img_sve_path=diretory_path / "train_eval_img" / f"epoch{epo}_ouput_img.png"
        plt.savefig(output_img_sve_path)        
        plt.close(output)
                

        if (epo) % config['save_freq'] == 0 or epo == config['epoch_num']:
            torch.save(net.state_dict(),'kd_epoch_{}_model.ckpt'.format(epo))
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import UNet
from torch.backends import cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from Dataset import split_data,collect_pairs,MedicalDataset
import tifffile
from evaluation import calculate_dice,batch_dice_score,visualize_and_calculate_dice
config=dict()
config['is_test']=False
config['dataset_split_rate']=3/7
config['seed']=42
file_dir=r"E:\Kaggle\src\Medical Image Segment\Data\DataPreprocess\kaggle_3m"
config['device']='cuda'
config['lr']=0.001
config['checkout']=r""
config['start_epoch']=1
config['epoch_num']=50
config['pring_freq_batch']=10
config['save_freq']=5
config['weights_save_path']=r"E:\Kaggle\src\Medical Image Segment\Data\weight"

def dice_loss(preds, targets, smooth=1e-5):
    """
    Dice Loss 函数（用于训练）。
    
    Args:
        preds: 预测结果 Tensor (Batch_Size, C, H, W) - 应该是概率值 [0, 1]
        targets: 真实标签 Tensor (Batch_Size, C, H, W) - 二值标签 {0, 1}
        smooth: 平滑项
        
    Returns:
        Tensor: Dice Loss (1 - Dice Coefficient)
    """
    batch_size = preds.size(0)
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


if __name__=="__main__":
    print("--- Getting start to Train ---")
    img_path_list,mask_path_list=collect_pairs(file_dir)
    train_path_pair,test_path_pair=split_data(img_path_list,mask_path_list,random_seed=config['seed'],split_rate=config['dataset_split_rate'])
    dataset=MedicalDataset(config,train_path_pair)
    cudnn.benchmark=True

    dataloader=DataLoader(dataset,shuffle=False,batch_size=64,num_workers=4)
    net=UNet(3)
    net.train()
    net.float().to(config["device"])

    criterion= nn.BCEWithLogitsLoss()

    optimizer=optim.Adam(net.parameters(),lr=config['lr'])
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout'],weights_only=True)) #防止恶意攻击
    for epo in range(config['start_epoch'],config['start_epoch']+config['epoch_num']+1):
        running_loss=0.0
        for i,(X,y) in enumerate(dataloader):
            
            optimizer.zero_grad()    #   确保初始梯度为0，不会积累
            
            
            X=X.to(config['device'])
            y=y.to(config['device'])
            output=net(X)
            loss1 = criterion(output, y)
            output=torch.sigmoid(output)
            loss2=dice_loss(output,y)

            loss=loss1+loss2
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.5)   #梯度裁剪
            optimizer.step()
            running_loss+=loss.item()
            if (i + 1) % config['pring_freq_batch'] == 0:
                avg_loss = running_loss / (i+1)
                output=output.cpu().detach().numpy()
                y=y.cpu().detach().numpy()
                dice=batch_dice_score(output,y)
                print(
                    '[ Epoch {:>3d} | Batch {:>5d} / {:>5d} ] Average Loss: {:.6f} Dice --->>> {:.4f}'.format(
                        epo, i + 1, len(dataloader), avg_loss,dice)
                    )
                
        
                

        if (epo) % config['save_freq'] == 0 or epo == config['epoch_num']:
            
            save_path=config['weights_save_path']
            save_path=Path(save_path)
            save_path=save_path / f"kd_epoch_{epo}_model.ckpt"
            torch.save(net.state_dict(),save_path)    

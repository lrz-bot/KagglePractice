import torch
from Dataset import collect_pairs,split_data,MedicalDataset
from torch.backends import cudnn
import evaluation
from torch.utils.data import DataLoader
from model import UNet
from pathlib import Path
config=dict()
config['is_test']=True
config['dataset_split_rate']=3/7
config['seed']=42
file_dir=r"E:\Kaggle\src\Medical Image Segment\Data\DataPreprocess\kaggle_3m"
config['device']='cuda'
config['checkout']=r"E:\Kaggle\src\Medical Image Segment\Data\bce_new_weight\kd_epoch_455_model.ckpt"
config['batch_size']=16
config['img_save_dir']=r"E:\Kaggle\src\Medical Image Segment\Data\res_bce_loss"
if __name__=="__main__":
    print("--- Getting start to Test ---")
    
    img_path_list,mask_path_list=collect_pairs(file_dir)
    train_path_pair,test_path_pair=split_data(img_path_list,mask_path_list,random_seed=config['seed'],split_rate=config['dataset_split_rate'])
    print(f"--- 测试集数量：{len(test_path_pair)} ---")
    dataset=MedicalDataset(config,test_path_pair)
    dataloader=DataLoader(dataset,batch_size=config['batch_size'])
    net=UNet(3)
    net.eval()
    net.to(config['device'])
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout'],weights_only=True)) #防止恶意攻击
    for i,(X,y) in enumerate(dataloader):
        X=X.to(config['device'])
        y=y.to(config['device'])
        with torch.no_grad():
             output=net(X)
        
        X=X.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        output=torch.sigmoid(output)
        output=output.cpu().detach().numpy()
        dice=evaluation.batch_dice_score(output,y)
        X=(X)/2+0.5
        print(f"--- 第{i}批次 Dice 均值--->>> {dice:.4f}")
        for j in range(X.shape[0]):
            save_path=Path(config['img_save_dir']) / f"batch_{i}_{j}_output.png"
            save_path=str(save_path)
            evaluation.visualize_and_calculate_dice(X[j,...],y[j,...],output[j,...],save_path)  
            print(f"--- batch_{i}_{j}_output 测试图片 输出完成 ---")
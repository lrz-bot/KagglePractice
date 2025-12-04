import matplotlib.pyplot as plt
import numpy as np




def plot_sample(x,y):
    """
    x:(96,96)
    y:(15,2)

    """
    
    fig,axis=plt.subplots(1,1,figsize=(12,9))
    axis.imshow(x,cmap='grey')
    axis.scatter(y[:,0],y[:,1],marker="x",s=50,color='red')
    return fig

def get_peak(heatmap_output,stride):
    """
    heatmap_output:(B,15,grid_x,grid_y)
    """
    B=heatmap_output.shape[0]
    keypoints=heatmap_output.shape[1]
    max_index=np.argmax(heatmap_output.reshape(B,keypoints,-1),axis=2)  #找到每个通道维度最大值的索引
    x_index=max_index//(heatmap_output.shape[3])   #形状为(B,15)
    y_index=max_index % (heatmap_output.shape[3])
    peak_coord=(np.stack([x_index,y_index],axis=2))  #此处得到的是在heatmap下的坐标
    peak_coord=peak_coord*stride+stride/2-0.5
    return peak_coord  #形状为(B,15,2)

def mse_eval(output,gt):
    """
    output:(B,15,2)
    gt:(B,15,2)
    """
    B=output.shape[0]
    keypoints=output.shape[1]
    gt_bool=np.isnan(gt)
    gt_bool=gt_bool+gt_bool[:,:,::-1]    #使得包含nan的行都含有True
    num_valid=np.sum((1-gt_bool))   #计算有效值数量
    gt=np.nan_to_num(gt)
    gt=gt*(1-gt_bool)   #确保所有空指标不对loss有贡献
    output=output*(1-gt_bool)
    
    mse=(np.sum(np.square((output-gt))))/(num_valid)

    return mse



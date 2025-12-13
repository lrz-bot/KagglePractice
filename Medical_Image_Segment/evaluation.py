import numpy as np
import matplotlib.pyplot as plt
import torch
def batch_dice_score(preds, targets, threshold=0.5):
    """
    计算批次的平均 Dice Score。
    
    Args:
        preds: 预测结果 (Batch_Size, C,H, W)
        targets: 真实标签 (Batch_Size, C, H, W)
        threshold: 二值化阈值
        
    Returns:
        float: 批次平均 Dice Score
    """
    preds = (preds > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    
    batch_size = preds.shape[0]
    preds = preds.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)
    
    intersection = (preds * targets).sum(axis=1)
    dice = (2. * intersection) / (preds.sum(axis=1) + targets.sum(axis=1) + 1e-5)
    
    return dice.mean()


def calculate_dice(pred, target, smooth=1e-5):
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def visualize_and_calculate_dice(image, mask, output, save_path):
    """
    可视化原始图像、真实Mask和预测Mask，并计算Dice系数。
    
    Args:
        image: 原始图像数据 (numpy array)
        mask: 真实标签 (numpy array)
        output: 模型输出 (numpy array)
        save_path: 图片保存路径
    """
    
    

    # 确保输入是 numpy 数组，如果是 Tensor 需要先转换
    # 这里假设传入的已经是 numpy array，如果不是可以在函数外处理或在这里添加类型检查
    
    # 处理预测结果，假设 output 需要经过阈值处理转为 0/1
    pred_mask = (output > 0.3).astype(np.float32)
    
    # 计算 Dice Score
    dice_score = calculate_dice(pred_mask, mask)
    print(f"Dice Score: {dice_score:.4f}")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    # 自动处理 (C, H, W) 到 (H, W, C) 的转换
    img_show = image
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        img_show = np.transpose(image, (1, 2, 0))
    # 如果是单通道灰度图，去掉通道维度以避免 matplotlib 警告
    if img_show.ndim == 3 and img_show.shape[2] == 1:
        img_show = img_show.squeeze(2)
    mask=np.squeeze(mask)  
    pred_mask=np.squeeze(pred_mask)  
    axes[0].imshow(img_show, cmap='gray' if img_show.ndim==2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 显示原始 Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    # 显示预测 Mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"Predicted Mask (Dice: {dice_score:.2f})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




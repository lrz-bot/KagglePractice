import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
from typing import Tuple, Optional
import tifffile


def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:

    h, w = image.shape[:2] 
    crop_h, crop_w = crop_size
    
    i_start = max(0, int(np.floor((h - crop_h) / 2)))
    j_start = max(0, int(np.floor((w - crop_w) / 2)))
    
    i_stop = i_start + crop_h
    j_stop = j_start + crop_w
    
    return image[i_start:i_stop, j_start:j_stop, ...]


class SimpleLGGPreprocessor:
    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def __call__(self, image_slice: np.ndarray, mask_slice: np.ndarray, 
                 is_mask: bool = False) -> np.ndarray:
        
        data = mask_slice.astype(np.float32) if is_mask else image_slice.astype(np.float32)
        
        # 1. 掩码二值化和孔洞填充 (仅对 mask 执行)
        if is_mask:
            data[data != 0] = 1
            # 填充孔洞 (逐通道处理)
            for c in range(data.shape[-1]):
                data[..., c] = binary_fill_holes(data[..., c] > 0.5).astype(np.float32)

        # 2. 重采样 (Resize)
        h, w = data.shape[:2]
        min_dim = min(h, w)
        
        if min_dim != self.target_size:
            scale = self.target_size / min_dim
            new_h, new_w = int(np.round(h * scale)), int(np.round(w * scale))
            new_shape = (new_h, new_w) + data.shape[2:]
            
            # 使用 bicubic 或 nearest 保持一致
            order = 0 if is_mask else 3 
            data = resize(data, new_shape, order=order, mode='edge', anti_aliasing=True, preserve_range=True)

        # 3. 居中裁剪到 256x256
        data = center_crop(data, (self.target_size, self.target_size))
        
        # 4. 修复负值 (仅对图像执行)
        if not is_mask:
            data[data < 0] = 0

        # 5. 简单的强度归一化 (仅对图像执行)
        if not is_mask:
            
            
            non_zero_data = data[data > 0] 
            current_min = np.min(non_zero_data) if non_zero_data.size > 0 else 0.0
            current_max = np.max(data)
            
            range_val = current_max - current_min
            
            if range_val > 0:
                # 裁剪并缩放到 [0, 1]
                data[data < current_min] = current_min
                data = (data - current_min) / range_val
            else:
                data[:] = 0.0 

        # 6. 转换为 8 位整数
        return img_as_ubyte(data)



    

image_path=r"E:\Kaggle\src\Medical Image Segment\Data\archive\lgg-mri-segmentation\kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_1.tif"
mask_path=r"E:\Kaggle\src\Medical Image Segment\Data\archive\lgg-mri-segmentation\kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_1_mask.tif"
image_np=tifffile.imread(image_path)
mask_np=tifffile.imread(mask_path)
image_save_path=r"E:\Kaggle\src\Medical Image Segment\Data\trial\img.tif"
mask_save_path=r"E:\Kaggle\src\Medical Image Segment\Data\trialimask.tif"
if len(mask_np.shape)==2:
    mask_np=mask_np[np.newaxis,:,:]

    
print("--- 即将预处理 ---")
simple_preprocessor = SimpleLGGPreprocessor()

print("--- 正在预处理 ---")
    
   
processed_img = simple_preprocessor(image_np, mask_np, is_mask=False)
    
processed_mask = simple_preprocessor(image_np, mask_np, is_mask=True)

print("--- 预处理完毕,准备开始保存 ---")
tifffile.imwrite(image_save_path,processed_img)
tifffile.imwrite(mask_save_path,processed_mask)
print("--- 保存完成 ---")

import numpy as np
import tifffile
import torch
import os
import concurrent.futures
from pathlib import Path
from typing import Tuple, Optional, List
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
 

# 确保导入 multiprocessing 以获取核心数


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
        
        if is_mask:
            data[data != 0] = 1
            for c in range(data.shape[-1]):
                data[..., c] = binary_fill_holes(data[..., c] > 0.5).astype(np.float32)

        h, w = data.shape[:2]
        min_dim = min(h, w)
        
        if min_dim != self.target_size:
            scale = self.target_size / min_dim
            new_h, new_w = int(np.round(h * scale)), int(np.round(w * scale))
            new_shape = (new_h, new_w) + data.shape[2:]
            
            order = 0 if is_mask else 3 
            data = resize(data, new_shape, order=order, mode='edge', anti_aliasing=True, preserve_range=True)

        data = center_crop(data, (self.target_size, self.target_size))
        
        if not is_mask:
            data[data < 0] = 0
        
            non_zero_data = data[data > 0] 
            current_min = np.min(non_zero_data) if non_zero_data.size > 0 else 0.0
            current_max = np.max(data)
            
            range_val = current_max - current_min
            
            if range_val > 0:
                data[data < current_min] = current_min
                data = (data - current_min) / range_val
            else:
                data[:] = 0.0 

        return img_as_ubyte(data)




def preprocess_and_overwrite_worker(task_tuple: Tuple[str, str]) -> str:
    """
    单个进程执行的函数。负责读取、预处理并覆盖原始图像/掩码文件。
    Args:
        task_tuple: (image_path, mask_path)
    Returns:
        处理结果状态字符串。
    """
    image_path, mask_path = task_tuple
    
    
    base_name = os.path.basename(image_path).replace('.tif', '')

    try:
        
        image_np = tifffile.imread(image_path)
        mask_np = tifffile.imread(mask_path)
        
        
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
        
        
        preprocessor = SimpleLGGPreprocessor()
        
        
        processed_img = preprocessor(image_np, mask_np, is_mask=False)
        processed_mask = preprocessor(image_np, mask_np, is_mask=True)
        
        
        tifffile.imwrite(image_path, processed_img)
        tifffile.imwrite(mask_path, processed_mask)
        
        return f"SUCCESS: Overwritten {base_name}"
        
    except Exception as e:
        return f"ERROR: Failed to overwrite {base_name}. Reason: {e}"


"""
------以下为参数部分
"""
data_dir=r'E:\Kaggle\src\Medical Image Segment\Data\DataPreprocess\kaggle_3m'
MAX_WORKERS = 4
"""
------以上为参数部分
"""

if __name__ == '__main__':
    # ------------------------------------------------
    #     执行并行处理
    # ------------------------------------------------
    data_dir=Path(data_dir)
    image_path_list=[]
    mask_path_list=[]
    for img in data_dir.rglob("*.tif"):
        p=img.stem
        if p.endswith("_mask"):
            continue

        mask_path_stem=f"{p}_mask"
        mask_path=img.with_stem(mask_path_stem)
        if mask_path.exists():
            image_path_list.append(str(img))
            mask_path_list.append(str(mask_path))

    img_mask_pairs=list(zip(image_path_list,mask_path_list))

    print(f"Warning: 正在对原始文件进行原地覆盖。使用 {MAX_WORKERS} 个进程...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        
        results = executor.map(preprocess_and_overwrite_worker, img_mask_pairs)
        
        
        for result in results:
            if "ERROR" in result:
                print(f"{result}")
            else:
                print(result)

    print("\n--- 并行原地预处理流程完成 ---")
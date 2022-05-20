import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize

import torch
from torchvision.utils import save_image

path = 'PH2_dataset/PH2 Dataset images'
files = os.listdir(path)

patch_size = 64

for img_name in files:
    img = imread(f'{path}/{img_name}/{img_name}_Dermoscopic_Image/{img_name}.bmp').astype(np.float32)
    mask = (imread(f'{path}/{img_name}/{img_name}_lesion/{img_name}_lesion.bmp', True) >= 128)
    
    img = resize(img, (256, 256))
    mask = resize(mask, (256, 256)).astype(np.float32)
    
    img_patches = torch.from_numpy(img).permute(2,0,1).unfold(dimension=1, size=patch_size, step=patch_size).unfold(dimension=2, size=patch_size, step=patch_size).permute(1,2,0,3,4)
    mask_patches = torch.from_numpy(mask).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size)
    
    path_save = os.path.join(f'PH2_dataset_patches/', f'{img_name}') 
    os.makedirs(path_save)
    path_save = os.path.join(f'PH2_dataset_patches/{img_name}/', f'{img_name}_Dermoscopic_Image') 
    os.makedirs(path_save)
    path_save = os.path.join(f'PH2_dataset_patches/{img_name}/', f'{img_name}_lesion') 
    os.makedirs(path_save)

    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            save_image(img_patches[i,j],f'PH2_dataset_patches/{img_name}/{img_name}_Dermoscopic_Image/{img_name}_patch_{i,j}.bmp', 'bmp', normalize = True)
            save_image(mask_patches[i,j],f'PH2_dataset_patches/{img_name}/{img_name}_lesion/{img_name}_patch_{i,j}_lesion.bmp', 'bmp')

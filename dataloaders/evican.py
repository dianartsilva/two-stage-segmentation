import numpy as np
import pandas as pd
import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset


class EVICAN(Dataset):

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        if self.fold == 'train':
            self.path_seg = 'EVICAN_dataset/Masks/EVICAN_train_masks/Cells'
            
        else:
            self.path_seg = 'EVICAN_dataset/Masks/EVICAN_val_masks/Cells'
        self.files = os.listdir(self.path_seg)
        self.transform = transform
            

    def __getitem__(self, i):
        f = self.files[i]
       
        if self.fold == 'train':
            img = imread(f'EVICAN_dataset/Images/EVICAN_train2019/{f}').astype(np.float32)
        else:
            img = imread(f'EVICAN_dataset/Images/EVICAN_val2019/{f}').astype(np.float32)

        seg = (imread(f'{self.path_seg}/{f}', True) >= 128).astype(np.float32)
        
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    evican = EVICAN('train')
    x, y = evican[1]
    print('x:', x.min(), x.max(), x.dtype, x.shape)
    print('y:', y.min(), y.max(), y.dtype, y.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(x.astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.show()

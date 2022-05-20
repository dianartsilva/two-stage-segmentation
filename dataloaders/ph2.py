import numpy as np
import pandas as pd
import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset

class PH2(Dataset):
    def __init__(self, fold, inputType = None, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        assert inputType in ['patches', None], f'inputType {inputType} must be patch or None'
        
        self.inputType = inputType
        self.path = f'datasets/PH2_dataset/PH2 Dataset images/'
        #print (self.path)
        
        self.files = os.listdir(self.path)
        self.transform = transform
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.files), len(self.files), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.files))]
            self.files = [self.files[i] for i in ix]
        else:
            ix = ix[int(0.70*len(self.files)):]
            self.files = [self.files[i] for i in ix]
        if inputType == 'patches':
            self.path = f'PH2_dataset_patches'
            self.files_p = [os.listdir(f'{self.path}/{name}/{name}_Dermoscopic_Image/') for name in self.files]
            #print('len files_p:', len(self.files_p))
            self.files = []
            [self.files.append(img_patches[i]) for img_patches in self.files_p for i in range (len(img_patches))] 
        #print('len files:', len(self.files))

    def __getitem__(self, i):
        f = self.files[i]
        #print(f)
        
        if self.inputType == 'patches':
            img = imread(f'{self.path}/{f[0:6]}/{f[0:6]}_Dermoscopic_Image/{f[:-4]}.bmp').astype(np.float32)
            seg = (imread(f'{self.path}/{f[0:6]}/{f[0:6]}_lesion/{f[:-4]}_lesion.bmp')>=128).astype(np.float32)
            seg = seg[:,:,0]
        else:
            img = imread(f'{self.path}/{f}/{f}_Dermoscopic_Image/{f}.bmp').astype(np.float32)
            seg = (imread(f'{self.path}/{f}/{f}_lesion/{f}_lesion.bmp', True) >= 128).astype(np.float32)

        #print('img:',type(img), img.dtype, img.shape)
        #print('seg:',type(seg), seg.dtype, seg.shape)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img_aug = d['image']
            seg_aug = d['mask']
            #print('imgph2:', img.min(), img.max(), img.dtype, img.shape)
            #print('segph2:', seg.min(), seg.max(), seg.dtype, seg.shape)
            return img, seg, img_aug, seg_aug

        return img, seg

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ph2 = PH2('train', 'patches')
    x, y = ph2[0]
    print('x:', x.min(), x.max(), x.dtype, x.shape)
    print('y:', y.min(), y.max(), y.dtype, y.shape)
   
    print('img:',type(x), x.dtype, x.shape)
    print('mask:',type(y), y.dtype, y.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(x.astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray', vmin=0, vmax=1)
    plt.show()

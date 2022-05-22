import numpy as np
import pandas as pd
import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset


class RETINA(Dataset):
    hi_size = 768
    pos_weight = 0.9096
    noutputs = 1
    nclasses = 2

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.path = 'datasets/RETINA_dataset'
        files = os.listdir(self.path)
        self.files = []
        for filename in files:    
            if filename[-9:] == 'image.png':
                self.files.append(filename[:-9])
        #print (self.files)
        self.transform = transform
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.files), len(self.files), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.files))]
            self.files = [self.files[i] for i in ix]
        else:
            ix = ix[int(0.70*len(self.files)):]
            self.files = [self.files[i] for i in ix]

    def __getitem__(self, i):
        f = self.files[i]
        img = imread(f'{self.path}/{f}image.png').astype(np.float32)
        seg = (imread(f'{self.path}/{f}mask.png', True) >= 128).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
            #print('imgph2:', img.min(), img.max(), img.dtype, img.shape)
            #print('segph2:', seg.min(), seg.max(), seg.dtype, seg.shape)
            
        #print('img:',type(img), img.dtype, img.shape)
        #print('seg:',type(seg), seg.dtype, seg.shape)
        return img, seg

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    retina = RETINA('train')
    x, y = retina[0]
    #print('x:', x.min(), x.max(), x.dtype, x.shape)
    #print('y:', y.min(), y.max(), y.dtype, y.shape)
   
    #x = tf.cast(x.permute(1,2,0),tf.uint8)
    print('img:',type(x), x.dtype, x.shape)
    print('mask:',type(y), y.dtype, y.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(x.astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.show()

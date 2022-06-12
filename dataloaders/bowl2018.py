import numpy as np
import pandas as pd
import os
from torchvision.io import read_image, ImageReadMode
import torch
from torch.utils.data import Dataset

class BOWL2018(Dataset):
    hi_size = 256
    pos_weight = 1  # 0.9764
    noutputs = 1
    nclasses = 2
    colors = 3
    can_rotate = True

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root = '/data/data-science-bowl-2018/stage1/train'
        self.dirs = os.listdir(self.root)
        self.transform = transform
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.dirs), len(self.dirs), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.dirs))]
            self.dirs = [self.dirs[i] for i in ix]
        else:
            ix = ix[int(0.70*len(self.dirs)):]
            self.dirs = [self.dirs[i] for i in ix]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        dir = self.dirs[i]
        img = read_image(os.path.join(self.root, dir, 'images', dir + '.png'), ImageReadMode.RGB).permute(1, 2, 0).numpy().astype(np.float32)
        masks = []
        for mask in os.listdir(os.path.join(self.root, dir, 'masks')):
            mask = os.path.join(self.root, dir, 'masks', mask)
            mask = read_image(mask, ImageReadMode.GRAY)[0].numpy() >= 128
            masks.append(mask)
        seg = np.clip(np.sum(masks, 0), 0, 1).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = BOWL2018('train')
    x, y = ds[1]
    print('x:', x.min(), x.max(), x.dtype, x.shape)
    print('y:', y.min(), y.max(), y.dtype, y.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.show()

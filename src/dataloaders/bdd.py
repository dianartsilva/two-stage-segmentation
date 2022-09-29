from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import os

class BDD(Dataset):
    hi_size = 768 #720
    pos_weight = 9.495455702930336
    noutputs = 1
    nclasses = 2
    colors = 3
    can_rotate = False

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        fold = fold if fold == 'train' else 'val'
        self.root_img = os.path.join('/data/bdd100k/images/10k', fold)
        self.root_seg = os.path.join('/data/bdd100k/labels/sem_seg/masks', fold)
        self.files = sorted(os.listdir(self.root_seg))
        self.transform = transform
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.files), len(self.files), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.files))]
        else:
            ix = ix[int(0.70*len(self.files)):]
        self.files = [self.files[i] for i in ix]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        img = imread(os.path.join(self.root_img, fname[:-3] + 'jpg')).astype(np.float32)
        seg = imread(os.path.join(self.root_seg, fname))
        seg[seg == 255] = 0
        #print(seg.max())
        # 11=pedestrian, 12=cyclist, 13=car, 14=van, 15=truck, 16=tram,
        # 17=motorcycle, 18=bicycle
        seg = np.logical_and(seg >= 13, seg <= 16).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = BDD('train')
    for i in range(len(ds)):
        img, seg = ds[i]
        plt.subplot(2, 1, 1)
        plt.imshow(img.astype(np.uint8))
        plt.subplot(2, 1, 2)
        plt.imshow(seg)
        plt.show()

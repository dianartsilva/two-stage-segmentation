from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import os

class KITTI(Dataset):
    hi_size = 512
    pos_weight = 14.070919091353261
    noutputs = 1
    nclasses = 2
    colors = 3
    can_rotate = False

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root_img = '/data/kitti/semantics/training/image_2'
        self.root_seg = '/data/kitti/semantics/training/semantic'
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
        img = imread(os.path.join(self.root_img, fname)).astype(np.float32)
        seg = imread(os.path.join(self.root_seg, fname))
        # 26-31 are cars, trucks and etc
        # 32=motorcycle, 33=bicycle, 25=cyclist, 24=pedestrian
        seg = np.logical_and(seg >= 26, seg <= 31).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = KITTI('train')
    img, seg = ds[0]
    plt.subplot(2, 1, 1)
    plt.imshow(img.astype(np.uint8))
    plt.subplot(2, 1, 2)
    plt.imshow(seg)
    plt.show()

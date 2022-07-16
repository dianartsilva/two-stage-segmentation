from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import os

def annotation(s, seg):
    starts = [int(x)-1 for x in s.split()[::2]]
    lengths = [int(y) for y in s.split()[1::2]]
    for s, l in zip(starts, lengths):
        seg.flat[s:s+l] = 1

class SARTORIUS(Dataset):
    hi_size = 512
    pos_weight = 8.539280150201735
    noutputs = 1
    nclasses = 2
    colors = 1
    can_rotate = True

    def __init__(self, fold, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root = '/data/sartorius-cell-instance-segmentation'
        self.transform = transform

        f = np.loadtxt(os.path.join(self.root, 'train.csv'), str, delimiter=',', skiprows=1, usecols=[0, 1, 2, 3])
        self.segs = {}
        for id, ann, w, h in f:
            if id not in self.segs:
                self.segs[id] = np.zeros((int(h), int(w)), np.float32)
            annotation(ann, self.segs[id])

        self.ids = list(self.segs)
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.ids), len(self.ids), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.ids))]
        else:
            ix = ix[int(0.70*len(self.ids)):]
        self.ids = [self.ids[i] for i in ix]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        img = imread(os.path.join(self.root, 'train', id + '.png'), True).astype(np.float32)
        seg = self.segs[id]
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = SARTORIUS('test')
    img, seg = ds[60]
    plt.subplot(2, 1, 1)
    plt.imshow(img.astype(np.uint8))
    plt.subplot(2, 1, 2)
    plt.imshow(seg)
    plt.show()

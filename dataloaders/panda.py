from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import os

class Panda(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = os.listdir(os.path.join(root, 'train_images'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        X = (imread(os.path.join(self.root, 'train_images', filename))/255).astype(np.float32)
        Y = imread(os.path.join(self.root, 'train_label_masks', filename[:-5] + '_mask.tiff'))[..., 0].astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return X, Y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = Panda('/data/prostate-cancer-grade-assessment')
    X, Y = ds[1]
    print('X:', X.dtype, X.shape, X.min(), X.max())
    print('Y:', Y.dtype, Y.shape, Y.min(), Y.max())
    plt.subplot(1, 2, 1)
    plt.imshow(X)
    plt.subplot(1, 2, 2)
    plt.imshow(Y, vmin=0, vmax=1, cmap='gray')
    plt.show()

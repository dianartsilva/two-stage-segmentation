class KITTI(Dataset):
    nclasses = 34

    def __init__(self, root, transform=None):
        self.root = root
        self.files = os.listdir(os.path.join(self.root, 'kitti', 'semantics', 'training', 'semantic'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        x = imread(os.path.join(self.root, 'kitti', 'semantics', 'training', 'image_2', filename)).astype(np.float32) / 255
        y = imread(os.path.join(self.root, 'kitti', 'semantics', 'training', 'semantic', filename))
        y = np.stack([y == i for i in range(self.nclasses)], 2)
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

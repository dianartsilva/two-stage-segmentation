import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchinfo import summary
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import losses
import importlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('fold', choices=['train', 'val', 'test'])
args = parser.parse_args()

################## MODEL ##################

model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)

model.load_state_dict(torch.load(f'results/ResNet50-{args.dataset}-patches-0.pth', map_location=torch.device('cpu')))
model = model.cuda()

################## LOAD DATASET ##################

i = importlib.import_module('dataloaders.' + args.dataset.lower())
ds = getattr(i, args.dataset.upper())

l = []
l += [A.Resize(ds.hi_size//8, ds.hi_size//8)]
l += [ToTensorV2()]
test_transforms = A.Compose(l)
testP_transforms = A.Compose([ToTensorV2()])

ts = ds(args.fold, transform=test_transforms)

################## EVALUATION ##################

ts = DataLoader(ts, batch_size=1, shuffle=False)
opt = optim.Adam(model.parameters())

loss_func = nn.BCEWithLogitsLoss()

model.eval()

path_save = os.path.join('results/ResNet50 Test Results/', f'{args.dataset}-{args.fold}')
os.makedirs(path_save)

numb = 0
avg_loss = 0
avg_dice = 0
for X, Y in ts:
    X = X.cuda()
    Y = Y[:, None, :,:].cuda()
#    print('X:',type(X), X.dtype, X.shape)
#    print('Y:',type(Y), Y.dtype, Y.shape)
    #print(X.device, Y.device)
    #print('model',next(model.parameters()).is_cuda)
    with torch.no_grad():
        Y_pred = model(X)['out']
    if ds.nclasses > 2:
        dice = 0
        loss = nn.functional.cross_entropy(Y_pred, Y)
    else:
        dice = losses.dice_score(torch.sigmoid(Y_pred), Y)
        loss = loss_func(Y_pred, Y) + (1-dice)
    avg_loss += float(loss) / len(ts)
    avg_dice += float(dice) / len(ts)
    
    if numb < 300:
   
        fig = plt.figure(figsize=(10,5))
        plt.ion()
        plt.subplot(1, 3, 1)
        plt.title('Image')
        X = X.cpu().permute(0,2,3,1).numpy()[0,:,:,:]
        plt.imshow(X.astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        Y = Y.cpu().numpy()[0,0,:,:]
        plt.imshow(Y, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('U-Net')
        Y_pred = Y_pred.cpu().numpy()[0,0,:,:]
        plt.imshow(Y_pred >= 0.5, cmap='gray')
        plt.show()
        plt.close(fig)
        fig.savefig(f'results/ResNet50 Test Results/{args.dataset}-{args.fold}/{args.dataset}_img{numb}.png',bbox_inches='tight', dpi=150)
    
    numb += 1
    
print(f'Model Loss: {avg_loss} - Dice: {avg_dice}')

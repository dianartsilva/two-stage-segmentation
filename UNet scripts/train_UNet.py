import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from time import time
import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
#print(summary(model))
model = model.to(device)


dataset = 'PH2-patches-64'

if dataset == 'PH2':
    from ph2 import PH2
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(180),
        A.RandomBrightnessContrast(0.1),
        A.Resize(282,282),
        A.RandomCrop(256, 256),
        ToTensorV2()
    ])
    tr = PH2('train', train_transforms)
    pos_weight = 0.6326
    
if dataset == 'PH2-patches-32':
    from ph2 import PH2
    patch_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(180),
        A.RandomBrightnessContrast(0.1),
        A.Resize(35,35),
        A.RandomCrop(32, 32),
        ToTensorV2()
    ])
    tr = PH2('train','patches', patch_transforms)
    pos_weight = 0.6860
        
if dataset == 'PH2-patches-64':
    from ph2 import PH2
    patch_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(180),
        A.RandomBrightnessContrast(0.1),
        A.Resize(70,70),
        A.RandomCrop(64, 64),
        ToTensorV2()
    ])
    tr = PH2('train','patches', patch_transforms)
    #pos_weight = 0.6860
    
if dataset == 'EVICAN':
    from evican import EVICAN
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(180),
        A.RandomBrightnessContrast(0.1),
        A.Resize(282,282),
        A.RandomCrop(256, 256),
        ToTensorV2()
    ])
    tr = EVICAN('train', train_transforms)
    pos_weight = 0.9764

if dataset == 'RETINA':
    from retina import RETINA
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(180),
        A.RandomBrightnessContrast(0.1),
        A.Resize(282,282),
        A.RandomCrop(256, 256),
        ToTensorV2()
    ])
    tr = RETINA('train', train_transforms)
    pos_weight = 0.9096

pos_weight = 1-torch.mean(torch.stack([y for x, y in tr]))
print('pos_weight:', pos_weight)

tr = DataLoader(tr, batch_size=64, shuffle=True)

learning_rate = 1e-5
opt = optim.Adam(model.parameters(),learning_rate)

# Dice and loss function
def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

loss_func = nn.BCELoss(reduction='none')

# Training the model

model.train()
print (f'\nTraining {dataset} dataset...\n')

EPOCHS = 300
total_time = 0
loss_values =[]
epoch_values = []
for epoch in range(EPOCHS):
    print(f'* Epoch {epoch+1} / {EPOCHS}')
    tic = time()
    avg_loss = 0
    avg_dice = 0
    for X, Y in tr:
        X = X.to(device)
        Y = Y[:, None, :,:].to(device)
#        print('X:', X.shape)
#        print('Y:', Y.shape)
        Y_pred = model(X)
#        print('Y_pred:', Y_pred.shape)
        dice = dice_score(Y_pred, Y)
        weights = (Y*pos_weight) + ((1-Y)*(1-pos_weight))
        loss = torch.mean(loss_func(Y_pred, Y) * weights)
        loss += (1-dice)
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        avg_dice += float(dice) / len(tr)
    toc = time()
    total_time += toc-tic
    loss_values.append(avg_loss)
    epoch_values.append(epoch+1)
    print(f'- Time: {toc-tic:.1f}s - Loss: {avg_loss} - Dice score: {avg_dice} ')

total_time = str(datetime.timedelta(seconds=round(total_time)))

torch.save(model.cpu().state_dict(), f'model-{dataset}.pth')

#print('loss values:', loss_values)
#print('total time:', total_time)

fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, loss_values)
plt.title(f'{dataset} - Training Time = {total_time} \n Learning rate = {learning_rate}')
plt.xlabel('EPOCH')
plt.ylabel('Loss')
plt.legend(loc='upper right')
fig.savefig(f'{dataset}_train_loss.png',bbox_inches='tight', dpi=150)

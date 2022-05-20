import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchinfo import summary
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from time import time
import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
#print(summary(model))
model = model.to(device)

DATASET = 'PH2'
SIZE_PATCHES = None  # None = no patches

l = [
    A.HorizontalFlip(),
    A.Rotate(180),
    A.RandomBrightnessContrast(0.1),
]
if SIZE_PATCHES == None:
    l += [A.RandomScale([1/8, 1/8], p=1)]
else:
    l += [A.RandomCrop(64, 64)]
l += [ToTensorV2()]
train_transforms = A.Compose(l)

if DATASET == 'PH2':
    from dataloaders.ph2 import PH2
    tr = PH2('train', None, train_transforms)
    pos_weight = 0.6326
if DATASET == 'EVICAN':
    from dataloaders.evican import EVICAN
    tr = EVICAN('train', train_transforms)
    pos_weight = 0.9764
if DATASET == 'RETINA':
    from dataloaders.retina import RETINA
    tr = RETINA('train', train_transforms)
    pos_weight = 0.9096

#pos_weight = 1-torch.mean(torch.stack([y for x, y in tr]))
#print('pos_weight:', pos_weight)

tr = DataLoader(tr, batch_size=1, shuffle=True, num_workers=8)

learning_rate = 1e-5
opt = optim.Adam(model.parameters(),learning_rate)

# Dice and loss function
def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

# Training the model

model.eval()  # desativar o BatchNorm FIXME
print (f'\nTraining {DATASET} {SIZE_PATCHES} dataset...\n')

EPOCHS = 300
total_time = 0
loss_values =[]
epoch_values = []
for epoch in range(EPOCHS):
    print(f'* Epoch {epoch+1} / {EPOCHS}')
    tic = time()
    avg_loss = 0
    avg_dice = 0
    for _, _, X, Y in tr:
        X = X.to(device)
        Y = Y[:, None, :,:].to(device)
        #print('X:', X.shape)
        #print('Y:', Y.shape)
        Y_pred = model(X)['out']
        #print('Y_pred:', Y_pred.shape)
        dice = dice_score(torch.sigmoid(Y_pred), Y)
        loss = loss_func(Y_pred, Y) + (1-dice)
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

torch.save(model.cpu().state_dict(), f'ResNet50-{DATASET}-{PATCH_SIZE}.pth')

#print('loss values:', loss_values)
#print('total time:', total_time)

fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, loss_values)
plt.title(f'{DATASET} {PATCH_SIZE} - Training Time = {total_time} \n Learning rate = {learning_rate}')
plt.xlabel('EPOCH')
plt.ylabel('Loss')
plt.legend(loc='upper right')
fig.savefig(f'ResNet50_{DATASET}_{PATCH_SIZE}_train_loss.png',bbox_inches='tight', dpi=150)

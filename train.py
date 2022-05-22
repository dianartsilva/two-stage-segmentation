import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
import datetime
import importlib
import losses

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('use_patches', choices=[0, 1], type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################## MODEL ##################

model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
model = model.to(device)

################## LOAD DATASET ##################

i = importlib.import_module('dataloaders.' + args.dataset.lower())
ds = getattr(i, args.dataset.upper())

l = [
    A.HorizontalFlip(),
    A.Rotate(180),
    A.RandomBrightnessContrast(0.1),
]
if args.use_patches: # low-resolution
    l += [A.RandomCrop(ds.hi_size//16, ds.hi_size//16)]
else:
    l += [
        A.Resize(ds.hi_size//8+ds.hi_size//20, ds.hi_size//8+ds.hi_size//20),
        A.RandomCrop(ds.hi_size//8, ds.hi_size//8),
    ]
l += [ToTensorV2()]
train_transforms = A.Compose(l)

tr = ds('train', transform=train_transforms)
pos_weight = ds.pos_weight

################## TRAINING ##################

tr = DataLoader(tr, batch_size=64, shuffle=True, num_workers=6)
opt = optim.Adam(model.parameters())

loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

# Training the model

model.train()
print (f'\nTraining {args.dataset} USEPATCHES={args.use_patches} dataset...\n')

EPOCHS = 300
if args.use_patches:
    EPOCHS *= 16

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
        #print('X:', X.shape)
        #print('Y:', Y.shape)
        Y_pred = model(X)['out']
        #print('Y_pred:', Y_pred.shape)
        if ds.nclasses > 2:
            dice = 0
            loss = nn.functional.cross_entropy(Y_pred, Y)
        else:
            dice = losses.dice_score(torch.sigmoid(Y_pred), Y)
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

torch.save(model.cpu().state_dict(), f'results/ResNet50-{args.dataset}-patches-{args.use_patches}.pth')

#print('loss values:', loss_values)
#print('total time:', total_time)

################## PLOT ##################

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, loss_values)
plt.title(f'{args.dataset} {args.use_patches} - Training Time = {total_time} \n Learning rate = 1e-03')
plt.xlabel('EPOCH')
plt.ylabel('Loss')
fig.savefig(f'results/ResNet50_{args.dataset}_patches_{args.use_patches}_train_loss.png',bbox_inches='tight', dpi=150)

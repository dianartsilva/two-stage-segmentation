import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torch import optim
from torchvision.ops.focal_loss import sigmoid_focal_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
from tqdm import tqdm
from torchinfo import summary
import datetime
import importlib
import losses
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model', choices=['deeplab', 'brain', 'ours'])
parser.add_argument('use_patches', choices=[0, 1], type=int)
parser.add_argument('--npatches', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################## LOAD DATASET ##################

i = importlib.import_module('dataloaders.' + args.dataset.lower())
ds = getattr(i, args.dataset.upper())

l = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(0.1),
]
if ds.can_rotate:
    l += [A.Rotate(180)]
if args.use_patches: # low-resolution
    l += [
        A.Resize(ds.hi_size, ds.hi_size),
        A.RandomCrop(ds.hi_size//args.npatches, ds.hi_size//args.npatches)
    ]
else:
    l += [
        A.Resize(ds.hi_size, ds.hi_size)
#        A.Resize(ds.hi_size//8+ds.hi_size//20, ds.hi_size//8+ds.hi_size//20),
#        A.RandomCrop(ds.hi_size//8, ds.hi_size//8),
    ]
l += [ToTensorV2()]
train_transforms = A.Compose(l)

tr = ds('train', transform=train_transforms)
pos_weight = ds.pos_weight


################## MODEL ##################

if args.model == 'deeplab':
    model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
elif args.model == 'brain':
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=ds.colors, out_channels=1, init_features=32, pretrained=True)
elif args.model == 'ours':
    from unet import UNet
    model = UNet(ds.colors)
model = model.to(device)

#summary(model, (10, ds.colors, ds.hi_size//args.npatches, ds.hi_size//args.npatches))

################## TRAINING ##################

tr = DataLoader(tr, batch_size=64, shuffle=True, num_workers=6)
learning_rate = 1e-4
opt = optim.Adam(model.parameters(), learning_rate)

#loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
loss_func = lambda ypred, y: sigmoid_focal_loss(ypred, y, reduction='mean')

# Training the model

model.train()
print (f'\nTraining {args.dataset} USEPATCHES={args.use_patches} dataset...\n')

EPOCHS = 200

if args.use_patches:
    EPOCHS *= 5

total_time = 0
loss_values =[]
epoch_values = []
acc_total = []
acc_0 = []
acc_1 = []
dice_values = []

for epoch in range(EPOCHS):
    print(f'* Epoch {epoch+1} / {EPOCHS}')
    tic = time()
    avg_loss = 0
    avg_dice = 0
    avg_acc = 0
    avg_acc0 = 0
    avg_acc1 = 0
    for X, Y in tqdm(tr):
        X = X.to(device)
        Y = Y[:, None, :,:].to(device)
        #print('X:', X.shape)
        #print('Y:', Y.shape)

        if args.model == 'deeplab' and ds.colors == 1:
            X = torch.cat((X, X, X), 1)
        Y_pred = model(X)
        if args.model == 'deeplab':
            Y_pred = Y_pred['out']
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
        acc = (Y >= 0.5) == (Y_pred >= 0.5)
        avg_acc += float(torch.mean(acc.float())) / len(tr)
        avg_acc0 += float(torch.sum((acc & (Y == 0)).float()) / torch.sum(Y == 0)) / len(tr)
        avg_acc1 += float(torch.sum((acc & (Y == 1)).float()) / torch.sum(Y == 1)) / len(tr)
    toc = time()
    total_time += toc-tic
    loss_values.append(avg_loss)
    epoch_values.append(epoch+1)
    acc_total.append(avg_acc)
    acc_0.append(avg_acc0)
    acc_1.append(avg_acc1)
    dice_values.append(avg_dice)
    print(f'- Time: {toc-tic:.1f}s - Loss: {avg_loss} - Dice score: {avg_dice} - Accuracy: {avg_acc} ({avg_acc0} vs {avg_acc1})')

total_time = str(datetime.timedelta(seconds=round(total_time)))

if args.use_patches:
    torch.save(model.cpu().state_dict(), f'results/{args.model}-{args.dataset}-patches-{args.use_patches}-{args.npatches}.pth')
else:
    torch.save(model.cpu().state_dict(), f'results/{args.model}-{args.dataset}-patches-{args.use_patches}.pth')

#print('loss values:', loss_values)
#print('total time:', total_time)

################## PLOT ##################

import matplotlib.pyplot as plt

path = f'results/[{args.dataset.upper()}] TRAIN'
if not os.path.exists(path):
    os.makedirs(path)


# Plotting Loss vs Epoch
fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, loss_values)
plt.title(f'{args.dataset} - {args.use_patches} - {args.npatches} - Training Time = {total_time} \n Learning rate = {learning_rate}')
plt.xlabel('EPOCH')
plt.ylabel('Loss')
fig.savefig(f'{path}/{args.model}_patches_{args.use_patches}_{args.npatches}_train_loss.png',bbox_inches='tight', dpi=150)

# Plotting Accuracy vs Epoch
fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, acc_total, label='Total Accuracy')
plt.plot(epoch_values, acc_0, label='0-pixels Accuracy')
plt.plot(epoch_values, acc_1, label='1-pixels Accuracy')
plt.xlabel('EPOCH')
plt.ylabel('Accuracy')
plt.legend()
fig.savefig(f'{path}/{args.model}_patches_{args.use_patches}_{args.npatches}_train_acc.png',bbox_inches='tight', dpi=150)

# Plotting Dice vs Epoch
fig = plt.figure(figsize=(10,5))
plt.plot(epoch_values, dice_values)
plt.xlabel('EPOCH')
plt.ylabel('Dice')
fig.savefig(f'{path}/{args.model}_patches_{args.use_patches}_{args.npatches}_train_dic.png',bbox_inches='tight', dpi=150)


f = open(f'{path}/{args.model}_patches_{args.use_patches}_{args.npatches}_results.txt', 'w')
print(f'Epoch: {epoch_values} \n', file=f)
print(f'Loss: {loss_values} \n', file=f)
print(f'Accuracy Total: {acc_total} \n', file=f)
print(f'Accuracy 0: {acc_0} \n', file=f)
print(f'Accuracy 1: {acc_1} \n', file=f)
print(f'Time: {total_time}', file=f)


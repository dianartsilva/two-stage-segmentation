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

from ph2 import PH2
from evican import EVICAN
from retina import RETINA

import patch
import os

dataset = 'PH2'
    
patch_size = 64

model_1 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model_1.load_state_dict(torch.load(f'model-{dataset}.pth', map_location=torch.device('cpu')))
model_1 = model_1.cuda()

model_2 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model_2.load_state_dict(torch.load(f'model-{dataset}-patches-{patch_size}.pth', map_location=torch.device('cpu')))
model_2 = model_2.cuda()

test_transforms = A.Compose([
    A.Resize(256, 256), 
    ToTensorV2()
])


if dataset == 'PH2':
    ts = PH2('train', None, test_transforms)
if dataset == 'EVICAN':
    ts = EVICAN('test', test_transforms)
if dataset == 'RETINA':
    ts = RETINA('test', test_transforms)
    
ts = DataLoader(ts, batch_size=1, shuffle=False)
opt = optim.Adam(model_1.parameters())

# Dice and loss function
def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

loss_func = nn.BCELoss()

model_1.eval()

numb = 0
loss_model1 = 0
dice_model1 = 0

loss_model2 = 0
dice_model2 = 0

loss_modelF = 0
dice_modelF = 0



# Model 1
for X, Y in ts:
    X1 = X.cuda()
    Y1 = Y[:, None, :,:].cuda()
    
    with torch.no_grad():
        Y_pred1 = model_1(X1)
    dice = dice_score(Y_pred1, Y1)
    loss = loss_func(Y_pred1, Y1) + (1-dice)
    loss_model1 += float(loss) / len(ts)
    dice_model1 += float(dice) / len(ts)

    path = os.path.join('proj', f'img{numb}') 
    os.makedirs(path)
    
    patch.visualize_model1(X1, Y1, Y_pred1, numb)
    
    # Patch Division
    X_patch = torch.squeeze(X1).unfold(dimension=1, size=patch_size, step=patch_size).unfold(dimension=2, size=patch_size, step=patch_size)
    X_patch = X_patch.permute(1,2,3,4,0)
    Y_patch = torch.squeeze(Y1).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size)
    Y_pred_patch = torch.squeeze(Y_pred1).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size).cpu()
    
    
    indices_val = patch.mean_patch(Y_pred_patch)
    patch.visualize_top10(X_patch, indices_val[0:10], dataset,'original',numb)
    patch.visualize_top10(Y_pred_patch, indices_val[0:10], dataset,'pred',numb)

    n = 0
    
    # Model 2
    model_2.eval()
    for [i,j] in indices_val[0:10]:

        X2 = X_patch[i,j].permute(2,0,1)[None,:,:,:]
        Y2 = Y_patch[i,j][None,None,:,:]
        
        with torch.no_grad():
            Y_pred2 = model_2(X2)
        
        dice = dice_score(Y_pred2, Y2)
        loss = loss_func(Y_pred2, Y2) + (1-dice)
        loss_model2 += float(loss) / (10*len(ts))
        dice_model2 += float(dice) / (10*len(ts))
        
        patch.visualize_seg(X2, Y2, Y_pred2, numb, n)
        
        Y_pred2 = torch.squeeze(Y_pred2)
        Y_pred_patch[i,j] = Y_pred2

        n += 1
     
    final_mask = patch.patches_concat(Y_pred_patch)[None, None, :, :].cuda()
    
    dice = dice_score(final_mask, Y1)
    loss = loss_func(final_mask, Y1) + (1-dice)
    loss_modelF += float(loss) / len(ts)
    dice_modelF += float(dice) / len(ts)
    
    # Save images
    final_mask = torch.squeeze(final_mask).cpu()
    patch.fig_save(final_mask, 'final_mask', numb)
    patch.visualize_models(Y1, Y_pred1, final_mask, numb)
    
    
    numb += 1
    
print(f'Model 1 Loss: {loss_model1} - Dice: {dice_model1}')
print(f'Model 2 Loss: {loss_model2} - Dice: {dice_model2}')
print(f'Model 1+2 Loss: {loss_modelF} - Dice: {dice_modelF}')


    


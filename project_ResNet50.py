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
from skimage.transform import resize

import patch
import os

dataset = 'panda'
    
patch_size = 64

model_1 = model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
model_1.load_state_dict(torch.load(f'ResNet50-{dataset}.pth', map_location=torch.device('cpu')))

model_1 = model_1.cuda()

model_2 = model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
model_2.load_state_dict(torch.load(f'ResNet50-{dataset}-patches-{patch_size}.pth', map_location=torch.device('cpu')))
model_2 = model_2.cuda()

test_transforms = A.Compose([
    A.RandomScale([1/8, 1/8], p=1),
    ToTensorV2()
])

if dataset == 'PH2':
    from ph2 import PH2
    ts = PH2('train', None, test_transforms)
if dataset == 'panda':
    from panda import Panda
    ts = Panda('/data/prostate-cancer-grade-assessment')
ts = DataLoader(ts, batch_size=1, shuffle=False)
opt = optim.Adam(model_1.parameters())

# Dice and loss function
def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

loss_func = nn.BCEWithLogitsLoss()

model_1.eval()

numb = 0
loss_model1 = 0
dice_model1 = 0

loss_model2 = 0
dice_model2 = 0

loss_modelF = 0
dice_modelF = 0

# Model 1
for X_hi, Y_hi, X_lo, Y_lo in ts:
    X_lo = X_lo.cuda()
    Y_lo = Y_lo[:, None, :,:].cuda()
    
    with torch.no_grad():
        Y_pred_lo = model_1(X_lo)['out']
    dice = dice_score(torch.sigmoid(Y_pred_lo), Y_lo)
    loss = loss_func(Y_pred_lo, Y_lo) + (1-dice)
    loss_model1 += float(loss) / len(ts)
    dice_model1 += float(dice) / len(ts)

    #path = os.path.join('proj1', f'img{numb}') 
    #os.makedirs(path)
    
    patch.visualize_model1(X_lo, Y_lo, Y_pred_lo, numb)
    
    # converter a segmentacao Y_pred_lo para Y_pred_hi para estar na mesma resolucao da X_hi
    Y_pred_hi = resize(Y_pred_lo, Y_pred_hi.shape[0])
    
    # Patch Division
    X_patch = torch.squeeze(X_hi).unfold(dimension=1, size=patch_size, step=patch_size).unfold(dimension=2, size=patch_size, step=patch_size)
    X_patch = X_patch.permute(1,2,3,4,0)
    Y_patch = torch.squeeze(Y_hi).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size)
    Y_pred_patch = torch.squeeze(Y_pred_hi).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size).cpu()
    
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
            Y_pred2 = model_2(X2)['out']
        
        dice = dice_score(torch.sigmoid(Y_pred2), Y2)
        loss = loss_func(Y_pred2, Y2) + (1-dice)
        loss_model2 += float(loss) / (10*len(ts))
        dice_model2 += float(dice) / (10*len(ts))
        
        patch.visualize_seg(X2, Y2, Y_pred2, numb, n)
        
        Y_pred2 = torch.squeeze(Y_pred2)
        Y_pred_patch[i,j] = Y_pred2

        n += 1
     
    final_mask = patch.patches_concat(Y_pred_patch)[None, None, :, :].cuda()
    
    dice = dice_score(torch.sigmoid(final_mask), Y_hi)
    loss = loss_func(final_mask, Y_hi) + (1-dice)
    loss_modelF += float(loss) / len(ts)
    dice_modelF += float(dice) / len(ts)
    
    # Save images
    final_mask = torch.squeeze(final_mask).cpu()
    patch.fig_save(final_mask, 'final_mask', numb)
    patch.visualize_models(Y_hi, Y_pred_hi, final_mask, numb)
    numb += 1
    
print(f'Model 1 Loss: {loss_model1} - Dice: {dice_model1}')
print(f'Model 2 Loss: {loss_model2} - Dice: {dice_model2}')
print(f'Model 1+2 Loss: {loss_modelF} - Dice: {dice_modelF}')

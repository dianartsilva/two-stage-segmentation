import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchinfo import summary
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset = 'PH2-patches-64'
setData = 'train'

model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)

model.load_state_dict(torch.load(f'ResNet50-{dataset}.pth', map_location=torch.device('cpu')))
model = model.cuda()

#print(summary(model))

test_transforms = A.Compose([
    A.Resize(256, 256), 
    ToTensorV2()
])

testP_transforms = A.Compose([
    #A.Resize(32, 32), 
    ToTensorV2()
])

if dataset == 'PH2':
    from ph2 import PH2
    ts = PH2(setData, None, test_transforms)
if dataset[0:11] == 'PH2-patches':
    from ph2 import PH2
    ts = PH2(setData, 'patches', testP_transforms)
if dataset == 'EVICAN':
    from evican import EVICAN
    ts = EVICAN(setData, test_transforms)
if dataset == 'RETINA':
    from retina import RETINA
    ts = RETINA(setData, test_transforms)
    
ts = DataLoader(ts, batch_size=1, shuffle=False)
opt = optim.Adam(model.parameters())

# Dice and loss function
def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

loss_func = nn.BCEWithLogitsLoss()

model.eval()

path_save = os.path.join('ResNet50 Test Results/', f'{dataset}-{setData}') 
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
    dice = dice_score(torch.sigmoid(Y_pred), Y)
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
        fig.savefig(f'ResNet50 Test Results/{dataset}-{setData}/{dataset}_img{numb}.png',bbox_inches='tight', dpi=150)
    
    numb += 1
    
print(f'Model Loss: {avg_loss} - Dice: {avg_dice}') 

    



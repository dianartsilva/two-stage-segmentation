import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchinfo import summary
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
import os
import importlib
import save, losses, patches_func

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

################## MODEL ##################

model_1 = model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
model_1.load_state_dict(torch.load(f'results/ResNet50-{args.dataset}-patches-0.pth', map_location=torch.device('cpu')))
model_1 = model_1.cuda()

model_2 = model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
model_2.load_state_dict(torch.load(f'results/ResNet50-{args.dataset}-patches-1.pth', map_location=torch.device('cpu')))
model_2 = model_2.cuda()

################## LOAD DATASET ##################

i = importlib.import_module('dataloaders.' + args.dataset.lower())
ds = getattr(i, args.dataset.upper())('test')
patch_size = ds.hi_size // 16

lo_transforms = A.Compose([
    A.Resize(ds.hi_size//8, ds.hi_size//8),
    ToTensorV2()
])
hi_transforms = A.Compose([
    A.Resize(ds.hi_size, ds.hi_size),
    ToTensorV2()
])

class LoHiTransform(Dataset):
    def __len__(self):
        return len(ds)

    def __getitem__(self, i):
        X, Y = ds[i]
        d = lo_transforms(image=X, mask=Y)
        X_lo = d['image']
        Y_lo = d['mask']
        d = hi_transforms(image=X, mask=Y)
        X_hi = d['image']
        Y_hi = d['mask']
        return X_lo, Y_lo[None], X_hi, Y_hi[None]

################## EVALUATION ##################

ts = LoHiTransform()
ts = DataLoader(ts, batch_size=1, shuffle=False, num_workers=6)
opt = optim.Adam(model_1.parameters())

loss_func = nn.BCEWithLogitsLoss()

model_1.eval()

path = f'results/Two Segmentation Results/final-seg'
if not os.path.exists(path):
    os.makedirs(path)


num = 0
loss_model1 = 0
dice_model1 = 0

loss_model2 = 0
dice_model2 = 0

loss_modelF = 0
dice_modelF = 0

# Model 1
for X_lo, Y_lo, X_hi, Y_hi in ts:
    X_lo = X_lo.cuda()
    Y_lo = Y_lo.cuda()
    X_hi = X_hi.cuda()
    Y_hi = Y_hi.cuda()

    with torch.no_grad():
        Y_pred_lo = model_1(X_lo)['out']
        Y_pred_hi_model1 = model_1(X_hi)['out']
    if ds.nclasses > 2:
        dice = 0
        loss = nn.functional.cross_entropy(Y_pred, Y)
    else:
        dice = losses.dice_score(torch.sigmoid(Y_pred_hi_model1), Y_hi)
        loss = loss_func(Y_pred_hi_model1, Y_hi) + (1-dice)
    loss_model1 += float(loss) / len(ts)
    dice_model1 += float(dice) / len(ts)

    path = f'results/Two Segmentation Results/img{num}'
    if not os.path.exists(path):
        os.makedirs(path)

    save.model1_result(X_lo, Y_lo, Y_pred_lo, num)

    # converter a segmentacao Y_pred_lo para Y_pred_hi para estar na mesma resolucao da X_hi
    Y_pred_hi = torch.tensor(resize(Y_pred_lo.cpu()[0, 0], Y_hi.shape[2:])[None, None])
    Y_pred_hi1 = torch.tensor(resize(Y_pred_lo.cpu()[0, 0], Y_hi.shape[2:])[None, None])

    # Patch Division
    X_patch = torch.squeeze(X_hi).unfold(dimension=1, size=patch_size, step=patch_size).unfold(dimension=2, size=patch_size, step=patch_size)
    X_patch = X_patch.permute(1,2,3,4,0)
    Y_patch = torch.squeeze(Y_hi).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size)
    _Y_pred_hi = Y_pred_hi if ds.nclasses == 2 else Y_pred_hi.max(1, keepdim=True)
    Y_pred_patch = torch.squeeze(_Y_pred_hi).unfold(dimension=0, size=patch_size, step=patch_size).unfold(dimension=1, size=patch_size, step=patch_size).cpu()

    indices_val = patches_func.min10_avg(Y_pred_patch)
    save.top10_result(Y_pred_patch, indices_val[0:10], args.dataset.upper(), 'pred', num)

    n = 0

    # Model 2
    model_2.eval()
    for [i,j] in indices_val[0:10]:

        X2 = X_patch[i,j].permute(2,0,1)[None,:,:,:].cuda()
        Y2 = Y_patch[i,j][None,None,:,:].cuda()
        
        with torch.no_grad():
            Y_pred2 = model_2(X2)['out']

        if ds.nclasses > 2:
            dice = 0
            loss = nn.functional.cross_entropy(Y_pred2, Y2)
        else:
            dice = losses.dice_score(torch.sigmoid(Y_pred2), Y2)
            loss = loss_func(Y_pred2, Y2) + (1-dice)
        loss_model2 += float(loss) / (10*len(ts))
        dice_model2 += float(dice) / (10*len(ts))
        
        save.model2_result(Y2, Y_pred_patch[i,j], Y_pred2, num, n)
        
        Y_pred2 = torch.squeeze(Y_pred2)
        Y_pred_patch[i,j] = Y_pred2

        n += 1

    final_mask = patches_func.concat(Y_pred_patch)[None, None, :, :].cuda()

    if ds.nclasses > 2:
        dice = 0
        loss = nn.functional.cross_entropy(final_mask, Y_hi)
    else:
        dice = losses.dice_score(torch.sigmoid(final_mask), Y_hi)
        loss = loss_func(final_mask, Y_hi) + (1-dice)
    loss_modelF += float(loss) / len(ts)
    dice_modelF += float(dice) / len(ts)

    # Save images
    final_mask = torch.squeeze(final_mask).cpu()
    save.fig(final_mask, 'final_mask', num)
    
    # FIXME: Não percebo porque é que, se tentar fazer save.TWOseg_result(Y_hi, Y_pred_hi, final_mask) o Y_pred_hi é igual à final mask (??) Para contornar isso tive de criar uma outra variável, a Y_pred_hi1 (se colocasse Y_pred_hi1 = Y_pred_hi no mesmo local onde declarei Y_pred_hi1 teria o mesmo problema de cima ?? )
    save.TWOseg_result(Y_hi, Y_pred_hi1, final_mask, patch_size, indices_val, num)
    num += 1

print(f'Model 1 Loss: {loss_model1} - Dice: {dice_model1}')
print(f'Model 2 Loss: {loss_model2} - Dice: {dice_model2}')
print(f'Model 1+2 Loss: {loss_modelF} - Dice: {dice_modelF}')

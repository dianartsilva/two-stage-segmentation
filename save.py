import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import torch
from torch import nn
import os

def patches_result(patches, dataset, mask_type, numb, path):
    '''Shows the patch division'''
    row = patches.shape[0]
    col = patches.shape[1]
    fig = plt.figure(figsize=(row, col))
    plt.ion()
    for i in range(row):
        for j in range(col):
            inp = patches[i][j].cpu()
            ax = fig.add_subplot(row, col, ((i*col)+j)+1, xticks=[], yticks=[])
            plt.imshow(inp, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/{dataset}_{mask_type}.png',bbox_inches='tight', dpi=150)
    
def top_result(patches, indices, dataset, mask_type, numb, path):
    '''Shows the patch division'''
    row = patches.shape[0]
    col = patches.shape[1]
    fig = plt.figure(figsize=(row, col))
    plt.ion()
    for i in range(row):
        for j in range(col):
            inp = patches[i][j].cpu()
            ax = fig.add_subplot(row, col, ((i*col)+j)+1, xticks=[], yticks=[])
            if [i,j] in indices:
                ax.patch.set_edgecolor('red')
                ax.patch.set_linewidth('7')
            if len(list(patches.shape)) == 5:
                #print('size:',inp.shape)
                plt.imshow(inp.numpy().astype(np.uint8), vmin = 0, vmax = 255)   
            else:
                plt.imshow(inp>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/{dataset}_{mask_type}_Top10.png',bbox_inches='tight', dpi=150)    

def model2_result(X, Y, Y_pred, numb, n, path):
    fig = plt.figure(figsize=(10,5))
    plt.ion()
    plt.subplot(1, 3, 1)
    plt.title('Ground Truth')
    X = torch.squeeze(X).cpu()
    plt.imshow(X >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title('ResNet50-patches-False')
    Y = torch.squeeze(Y).cpu()
    plt.imshow(Y>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title('ResNet50-patches-True')
    Y_pred = torch.squeeze(Y_pred).cpu()
    plt.imshow(Y_pred >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/patch{n}.png',bbox_inches='tight', dpi=150)

def fig(img, name, numb, path):
    fig = plt.figure(figsize=(10,5))
    plt.imshow(img >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/{name}.png',bbox_inches='tight', dpi=150)
    
def TWOseg_result(Y, Y_pred1, Y_pred2, patch_size, top, numb, path, npatches):
    fig = plt.figure(figsize=(10,5))
    plt.ion()
    plt.subplot(1, 3, 1)
    plt.title('Ground Truth')
    Y = torch.squeeze(Y).cpu()
    plt.imshow(Y>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title('STAGE 1 SEG')
    Y_pred1 = torch.squeeze(Y_pred1).cpu()
    plt.imshow(Y_pred1>=0.5, cmap='gray', vmin=0, vmax=1)
    ax = plt.gca()
    for i,j in top:
        rect = ptc.Rectangle((j*patch_size, i*patch_size), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.subplot(1, 3, 3)
    plt.title('STAGE 2 SEG')
    Y_pred2 = torch.squeeze(Y_pred2).cpu()
    plt.imshow(Y_pred2>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/models.png',bbox_inches='tight', dpi=150)
    fig.savefig(f'results/Two Segmentation Results - {npatches}/final-seg/img{numb}-two-seg.png',bbox_inches='tight', dpi=150)
    
def model1_result(X, Y, Y_pred, numb, path):
    fig = plt.figure(figsize=(10,5))
    plt.ion()
    plt.subplot(1, 3, 1)
    plt.title('Image')
    X = X.cpu().permute(0,2,3,1).numpy()[0,:,:,:]
    plt.imshow(X.astype(np.uint8))
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    Y = torch.squeeze(Y).cpu()
    plt.imshow(Y>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title('ResNet50-patches-False')
    Y_pred = torch.squeeze(Y_pred).cpu()
    plt.imshow(Y_pred >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'{path}/model1.png',bbox_inches='tight', dpi=150)

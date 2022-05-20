import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn


def patches_save(patches, dataset, mask_type, numb):
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
    fig.savefig(f'proj1/img{numb}/{dataset}_{mask_type}.png',bbox_inches='tight', dpi=150)


def mean_patch(patches):
    mean_values = []
    ind_values = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            mean_values.append(abs(torch.mean(patches[i][j])-0.5)) 
            ind_values.append([i,j])       
    argmean_sorted = np.argsort(mean_values, axis=0)
    ind_min10 = [ind_values[i] for i in argmean_sorted[0:10]]
    return ind_min10
    
def visualize_top10(patches, indices, dataset, mask_type, numb):
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
    fig.savefig(f'proj1/img{numb}/{dataset}_{mask_type}.png',bbox_inches='tight', dpi=150)    


def visualize_seg (X, Y, Y_pred, numb, n):
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
    plt.title('U-Net')
    Y_pred = torch.squeeze(Y_pred).cpu()
    plt.imshow(Y_pred >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'proj1/img{numb}/patch{n}.png',bbox_inches='tight', dpi=150)

def fig_save(img, name, numb):
    fig = plt.figure(figsize=(10,5))
    plt.imshow(img >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'proj1/img{numb}/{name}.png',bbox_inches='tight', dpi=150)
    
def patches_concat(patches):     
    col_concat = torch.tensor([])
    final_mask = torch.tensor([])
    
    for lin in range (patches.shape[0]):
        for col in range (patches.shape[1]):
            col_concat = torch.cat((col_concat,patches[lin,col]), dim=1)
            
        final_mask = torch.cat((final_mask,col_concat), dim=0)
        col_concat = torch.tensor([])
        
    return final_mask
        
def visualize_models(Y, Y_pred1, Y_pred2, numb):
    fig = plt.figure(figsize=(10,5))
    plt.ion()
    plt.subplot(1, 3, 1)
    plt.title('Ground Truth')
    Y = torch.squeeze(Y).cpu()
    plt.imshow(Y>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title('U-Net: total image')
    Y_pred1 = torch.squeeze(Y_pred1).cpu()
    plt.imshow(Y_pred1>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title('U-Net: patches')
    Y_pred2 = torch.squeeze(Y_pred2).cpu()
    plt.imshow(Y_pred2>=0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'proj1/img{numb}/models.png',bbox_inches='tight', dpi=150)
    
def visualize_model1 (X, Y, Y_pred, numb):
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
    plt.title('U-Net')
    Y_pred = torch.squeeze(Y_pred).cpu()
    plt.imshow(Y_pred >= 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close(fig)
    fig.savefig(f'proj1/img{numb}/model1.png',bbox_inches='tight', dpi=150)

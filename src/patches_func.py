import torch
import numpy as np


def avg_sorted(patches):
    ind_values = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            mean_val = abs(torch.mean(patches[i][j])-0.5)
            if mean_val <= 0.25:
                ind_values.append([i,j])   
    return ind_values

def min_top_avg(patches, ntop):
    mean_values = []
    ind_values = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            mean_values.append(abs(torch.mean(patches[i][j])-0.5)) 
            ind_values.append([i,j])       
    argmean_sorted = np.argsort(mean_values, axis=0)
    ind_min10 = [ind_values[i] for i in argmean_sorted[0:ntop]]
    return ind_min10

def concat(patches):     
    col_concat = torch.tensor([])
    final_mask = torch.tensor([])
    
    for lin in range (patches.shape[0]):
        for col in range (patches.shape[1]):
            col_concat = torch.cat((col_concat,patches[lin,col]), dim=1)
            
        final_mask = torch.cat((final_mask,col_concat), dim=0)
        col_concat = torch.tensor([])
        
    return final_mask

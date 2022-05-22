import torch

def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

def BCE_Dice_Loss(pos_weight):
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def f(y_pred, y_true):
        bce = bce_loss(y_pred, y_true)
        dice = 1 - dice_score(torch.sigmoid(y_pred), y_true)
        return bce + dice
    return f

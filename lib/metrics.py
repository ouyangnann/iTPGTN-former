import numpy as np
import torch


def masked_mse_torch(preds, labels, null_val=0):
    labels[labels<1e-3] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= mask.mean()
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss.mean()


def masked_mae_torch(preds, labels):
    labels[labels < 1e-3] = 0
    mask = (labels != 0).float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    ## trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_torch(preds, labels):
    mse = masked_mse_torch(preds, labels)
    return torch.sqrt(mse)


def masked_mape_torch(preds, labels, null_val= 0):
    labels[labels<1e-3] = 0
    if np.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mean_mask = torch.mean(mask) 
    mask /= mean_mask 
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

"""
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_torch(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss




def masked_mape_np(preds, labels, null_val=np.nan, mask_val=np.nan, epsilon=1e-10):
    # 处理极小值
    labels = np.where(np.abs(labels) < epsilon, 0, labels)

    # 创建遮罩
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = labels != null_val
    if not np.isnan(mask_val):
        mask &= labels >= mask_val

    # 计算损失
    loss = np.abs((preds - labels) / (labels + epsilon))
    loss = np.where(mask, loss, np.nan)

    # 排除NaN值
    valid_loss = loss[~np.isnan(loss)]

    # 计算平均损失
    if valid_loss.size == 0:
        return np.nan  # 避免除以零
    return np.mean(valid_loss)"""


def calculate_metrics(df_pred, df_test, null_val= 1e-3):
    #mae = masked_mae_np(preds=df_pred, labels=df_test)
    mape = masked_mape_torch(preds=df_pred, labels=df_test)
    rmse = masked_rmse_torch(preds=df_pred, labels=df_test)
    return rmse, mape
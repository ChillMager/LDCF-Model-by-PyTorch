"""
Created on 2018/10/21 by Chunhui Yin.
Reproduced on 2026/4/3 by Nan Sun.
Converted to PyTorch.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch


"""def evaluate_loader(model, test_loader):
    #Evaluate using PyTorch DataLoader
    model.eval()
    all_preds = []
    all_targets = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for (user_id, user_geo, service_id, service_geo), qos in test_loader:
            user_id = user_id.to(device)
            user_geo = user_geo.to(device)
            service_id = service_id.to(device)
            service_geo = service_geo.to(device)
            qos = qos.to(device)
            pred = model(user_id, user_geo, service_id, service_geo)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(qos.cpu().numpy().flatten())
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return mae, rmse"""


def evaluate_loader(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    device = next(model.parameters()).device

    # 检查是否需要反变换
    use_log = False
    if hasattr(test_loader.dataset, 'use_log'):
        use_log = test_loader.dataset.use_log

    with torch.no_grad():
        for (user_id, user_geo, service_id, service_geo), qos in test_loader:
            user_id = user_id.to(device)
            user_geo = user_geo.to(device)
            service_id = service_id.to(device)
            service_geo = service_geo.to(device)
            qos = qos.to(device)
            pred = model(user_id, user_geo, service_id, service_geo)

            # 如果之前做了对数变换，这里要反变换回原始 TP 值
            if use_log:
                pred = torch.expm1(pred)
                qos = torch.expm1(qos)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(qos.cpu().numpy().flatten())

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return mae, rmse


def saveResult(resultPath, dataType, density, results, metrics):
    """Save evaluation results (results: numpy array of shape [epochs, len(metrics)])"""
    import os
    os.makedirs(resultPath, exist_ok=True)
    if density:
        fileID = open(f'{resultPath}/{dataType}_result_{density:.2f}.txt', 'w')
    else:
        fileID = open(f'{resultPath}/{dataType}_result.txt', 'w')
    fileID.write('Metric: ')
    for metric in metrics:
        fileID.write(f'| {metric}\t')
    avgResult = np.average(results, axis=0)
    fileID.write('\nAvg:\t')
    np.savetxt(fileID, np.matrix(avgResult), fmt='%.4f', delimiter='\t')
    minResult = np.min(results, axis=0)
    fileID.write('Min:\t')
    np.savetxt(fileID, np.matrix(minResult), fmt='%.4f', delimiter='\t')
    fileID.write('\n==================================\n')
    fileID.write(f'Detailed results for {results.shape[0]} epochs:\n')
    np.savetxt(fileID, results, fmt='%.4f', delimiter='\t')
    fileID.close()
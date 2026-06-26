"""
Created on 2018/10/21 by Chunhui Yin.
Reproduced on 2026/4/3 by Nan Sun.
Converted to PyTorch.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class WSDreamDataset(Dataset):
    """把数据变成 PyTorch 张量（Tensor）"""
    """def __init__(self, file_path, data_type, density, mode='train'):
        if mode == 'train':
            filename = f'{file_path}{data_type}_train_{density:.2f}.txt'
        else:
            filename = f'{file_path}{data_type}_test_{density:.2f}.txt'
        data = pd.read_csv(filename, sep='\t')
        # 一次性全部转 tensor
        self.user_id = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.service_id = torch.tensor(data.iloc[:, 1].values, dtype=torch.long)
        self.qos = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32)
        self.user_geo = torch.tensor(data.iloc[:, [3, 4]].values, dtype=torch.long)
        self.service_geo = torch.tensor(data.iloc[:, [5, 6]].values, dtype=torch.long)"""

    def __init__(self, file_path, data_type, density, mode='train'):
        if mode == 'train':
            filename = f'{file_path}{data_type}_train_{density:.2f}.txt'
        else:
            filename = f'{file_path}{data_type}_test_{density:.2f}.txt'
        data = pd.read_csv(filename, sep='\t')
        # Pre-convert columns to tensors for fast indexing
        self.user_id = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.service_id = torch.tensor(data.iloc[:, 1].values, dtype=torch.long)

        # 对 TP 做对数变换
        qos_raw = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32)
        if data_type == 'tp':
            self.qos = torch.log1p(qos_raw)  # log(1 + TP)
            self.use_log = True
        else:
            self.qos = qos_raw
            self.use_log = False

        self.user_geo = torch.tensor(data.iloc[:, [3, 4]].values, dtype=torch.long)
        self.service_geo = torch.tensor(data.iloc[:, [5, 6]].values, dtype=torch.long)

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        return (self.user_id[idx], self.user_geo[idx],
                self.service_id[idx], self.service_geo[idx]), self.qos[idx]


class DataSet:
    """创建 DataLoader（批次加载器），并告诉模型有多少用户、多少服务"""
    def __init__(self, dataType, density):
        self.dataType = dataType
        self.density = density
        self.filePath = './Data/WSDream/Dataset#1/'
        # 读取原始数据，统计用户总数、服务总数
        origin = pd.read_csv(f'{self.filePath}{dataType}_origin.txt', sep='\t')
        self.num_users = origin.iloc[:, 0].max() + 1
        self.num_services = origin.iloc[:, 1].max() + 1
        print(f"userNum={self.num_users} | serviceNum={self.num_services} | density={density:.2f} | dataType={dataType}")

    def get_dataloaders(self, batch_size=256):
        train_ds = WSDreamDataset(self.filePath, self.dataType, self.density, 'train')
        test_ds = WSDreamDataset(self.filePath, self.dataType, self.density, 'test')
        # 训练集打乱, 测试集不打乱
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
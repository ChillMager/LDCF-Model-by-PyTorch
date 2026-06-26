"""
Created on 2018/10/21 by Chunhui Yin(yinchunhui.ahu@gmail.com).
Description:Loading the data.

"""
import sys
from time import time
import numpy as np
import pandas as pd


class DataSet(object):
    def __init__(self, dataType, density):
        self.dataType, self.density = dataType, density # re/tp, [0.05~0.30]
        self.data, self.shape = self.getData()
        # [用户ID, 服务ID, QoS值, 用户国家, 用户ASN, 服务国家, 服务ASN],
        # (用户总数339, 服务总数5825)
        self.train, self.test = self.getTrainTest() # 训练集, 测试集

    def getData(self):
        self.start = time()
        sys.stdout.write('\rLoading data...')
        if self.dataType == 'rt' or self.dataType == 'tp':
            self.filePath = './Data/WSDream/Dataset#1/'
        else:
            sys.stdout.write('\rData type error.')
            sys.exit()
        data = pd.read_csv(self.filePath + '%s_origin.txt' % self.dataType, sep='\t')
        return data, [data.iloc[:, 0].max() + 1, data.iloc[:, 1].max() + 1]

    def getTrainTest(self):
        train = pd.read_csv(self.filePath + '%s_train_%.2f.txt' % (self.dataType, self.density), sep='\t')
        test = pd.read_csv(self.filePath + '%s_test_%.2f.txt' % (self.dataType, self.density), sep='\t')
        sys.stdout.write("\rLoading completes.[%.2fs] userNum=%d | serviceNum=%d | density=%.2f | dataType=%s\n"
                         % (time() - self.start, self.shape[0], self.shape[1], self.density, self.dataType))
        return train, test

    def getTrainInstance(self, data):
        userID = np.array(data.iloc[:, 0])
        userGeo = np.array(data.iloc[:, [3, 4]])
        serviceID = np.array(data.iloc[:, 1])
        serviceGeo = np.array(data.iloc[:, [5, 6]])
        QoS = np.array(data.iloc[:, 2])
        return [userID, userGeo, serviceID, serviceGeo], QoS
    """
        用户ID → 第0列
        用户位置（国家+ASN）→ 第3、4列 （拼接）
        服务ID → 第1列
        服务位置（国家+ASN）→ 第5、6列 （拼接）
        QoS 真实值 → 第2列
    """

    def getTestInstance(self, data):
        userID = np.array(data.iloc[:, 0])
        userGeo = np.array(data.iloc[:, [3, 4]])
        serviceID = np.array(data.iloc[:, 1])
        serviceGeo = np.array(data.iloc[:, [5, 6]])
        QoS = np.array(data.iloc[:, 2])
        return [userID, userGeo, serviceID, serviceGeo], QoS

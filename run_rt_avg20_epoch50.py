"""
Created on 2018/10/21 by Chunhui Yin.
Reproduced on 2026/4/3 by Nan Sun.
Converted to PyTorch.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DataSet import DataSet
from Evaluator import evaluate_loader, saveResult


def huber_loss(pred, target, delta=1.0):
    error = target - pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()


class LDCFModel(nn.Module):
    def __init__(self, num_users, num_services, layers, use_cosine=True):
        super(LDCFModel, self).__init__()
        self.num_users = num_users
        self.num_services = num_services
        self.layers = layers
        self.use_cosine = use_cosine
        #
        self.emb_dim = layers[0] // 4

        self.user_id_embed = nn.Embedding(num_users, self.emb_dim)
        self.user_loc_embed = nn.Embedding(num_users, self.emb_dim)
        self.service_id_embed = nn.Embedding(num_services, self.emb_dim)
        self.service_loc_embed = nn.Embedding(num_services, self.emb_dim)
        # mlp_input_dim = 16 + 32 + 16 + 32 = 96
        mlp_input_dim = self.emb_dim + (self.emb_dim * 2) + self.emb_dim + (self.emb_dim * 2)
        mlp_layers = []
        current_dim = mlp_input_dim
        # range(1, 5-1) → range(1,4) → 32, 16 , 8
        for i in range(1, len(layers) - 1):
            # 第一次: 96 → 32 , ... , 16 → 8
            mlp_layers.append(nn.Linear(current_dim, layers[i]))
            mlp_layers.append(nn.ReLU())
            current_dim = layers[i]
        self.mlp = nn.Sequential(*mlp_layers)
        """
        [ Linear, ReLU, Linear, ReLU, Linear, ReLU ]
        mlp_layers = []
        mlp_layers.append(nn.Linear(96, 32))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(32, 16))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(16, 8))
        mlp_layers.append(nn.ReLU())
        """
        self.final_layer = nn.Linear(layers[-2] + 1, layers[-1])
        # 初始化所有层的权重
        self._init_weights()

    def _init_weights(self):
        """ 初始化权重 """
        # 遍历 embedding, linear, relu, sequential
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_id, user_loc, service_id, service_loc):
        user_id_vec = self.user_id_embed(user_id) # 1 → 16
        user_loc_vec = self.user_loc_embed(user_loc).view(user_loc.size(0), -1) # 2 → 32
        """
        [国家的16维]
        [ASN的16维] → [国家16维 + ASN16维] 
        """
        service_id_vec = self.service_id_embed(service_id) # 1 → 16
        service_loc_vec = self.service_loc_embed(service_loc).view(service_loc.size(0), -1) # 2 → 32
        mlp_input = torch.cat([user_id_vec, user_loc_vec, service_id_vec, service_loc_vec], dim=1)
        mlp_output = self.mlp(mlp_input) # 96维向量

        # Adaptive Corrector（自适应校正模块）
        user_loc_emb = self.user_loc_embed(user_loc)
        service_loc_emb = self.service_loc_embed(service_loc)
        """
        把用户/服务的 国家、ASN 两个数字 变成 两个 16 维向量
        """
        user_loc_avg = user_loc_emb.mean(dim=1)
        service_loc_avg = service_loc_emb.mean(dim=1)
        """
        dim=0：256 → 一批有多少条数据（不动）
        dim=1：2 → 每个数据有2 个特征（国家、ASN）← 要平均的就是这个维度！
        dim=2：16 → 每个特征是 16 维向量
        ( 国家的16维向量 + ASN的16维向量 ) ÷ 2 = 得到一个新的 16维向量
        """
        sim = torch.cosine_similarity(user_loc_avg, service_loc_avg, dim=1, eps=1e-8).unsqueeze(1)
        final_input = torch.cat([mlp_output, sim], dim=1)
        prediction = self.final_layer(final_input)
        return prediction


class LDCF:
    def __init__(self, args, density, run_id=0):
        self.args = args
        self.run_id = run_id
        self.dataset = DataSet(args.dataType, density)
        self.train_loader, self.test_loader = self.dataset.get_dataloaders(args.batchSize)
        self.num_users = self.dataset.num_users
        self.num_services = self.dataset.num_services

        self.epochNum = args.epochNum
        self.lr = args.lr
        self.decay = args.decay
        self.verbose = args.verbose
        self.store = args.store
        self.modelPath = args.modelPath
        self.resultPath = args.resultPath

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LDCFModel(self.num_users, self.num_services, args.layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

    def run(self):
        # 先做一次初始评估
        mae, rmse = evaluate_loader(self.model, self.test_loader)
        best_mae, best_rmse, best_epoch = mae, rmse, -1
        # 建一个表格，用来存每一轮的 MAE 和 RMSE , 行数 = 训练多少轮（epochNum）, 列数 = 2（MAE、RMSE）
        evalResults = np.zeros((self.epochNum, 2))
        print(f'Initial: MAE = {mae:.4f} | RMSE = {rmse:.4f}')

        for epoch in range(self.epochNum):
            self.model.train()
            total_loss = 0.0
            for (user_id, user_geo, service_id, service_geo), qos in self.train_loader:
                user_id = user_id.to(self.device)
                user_geo = user_geo.to(self.device)
                service_id = service_id.to(self.device)
                service_geo = service_geo.to(self.device)
                qos = qos.to(self.device)
                # 清空上一轮的梯度
                self.optimizer.zero_grad()
                pred = self.model(user_id, user_geo, service_id, service_geo)
                loss = huber_loss(pred.squeeze(), qos)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(user_id)

            avg_loss = total_loss / len(self.train_loader.dataset)

            if epoch % self.verbose == 0:
                mae, rmse = evaluate_loader(self.model, self.test_loader)
                if mae < best_mae:
                    best_mae, best_rmse, best_epoch = mae, rmse, epoch
                    if self.store:
                        self.saveModel()
                evalResults[epoch, :] = [mae, rmse]
                print(f'Epoch {epoch:3d}: MAE = {mae:.4f} | RMSE = {rmse:.4f} | Loss = {avg_loss:.4f}')

        print(f'Best at epoch {best_epoch}: MAE = {best_mae:.4f} | RMSE = {best_rmse:.4f}')
        if self.store:
            result_path_with_run = f'{self.resultPath}/run_{self.run_id}'
            saveResult(result_path_with_run, self.dataset.dataType, self.dataset.density, evalResults, ['MAE', 'RMSE'])
        return best_mae, best_rmse

    def saveModel(self):
        os.makedirs(self.modelPath, exist_ok=True)
        torch.save(self.model.state_dict(),
                   f'{self.modelPath}/{self.dataset.dataType}_{self.dataset.density:.2f}_{self.args.layers}_run{self.run_id}.pth')


def train_multiple_runs(args, density, num_runs=20):
    """同一个配置跑 20 次,取 20 次结果的平均值和标准差"""
    all_mae, all_rmse = [], []
    print(f"\n{'='*70}\nTraining density = {density} for {num_runs} runs, {args.epochNum} epochs each\n{'='*70}")
    for run in range(num_runs):
        print(f"\n{'='*50}\nRun {run+1}/{num_runs}\n{'='*50}")
        seed = run * 42 + int(density * 10000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        ld = LDCF(args, density, run_id=run)
        best_mae, best_rmse = ld.run()
        all_mae.append(best_mae)
        all_rmse.append(best_rmse)
        print(f"Run {run+1} best: MAE={best_mae:.4f}, RMSE={best_rmse:.4f}")

    all_mae = np.array(all_mae)
    all_rmse = np.array(all_rmse)
    avg_mae, std_mae = np.mean(all_mae), np.std(all_mae)
    avg_rmse, std_rmse = np.mean(all_rmse), np.std(all_rmse)
    print(f"\n{'='*70}\nFINAL RESULTS FOR DENSITY = {density} ({num_runs} runs, {args.epochNum} epochs)\n{'='*70}")
    print(f"MAE:  {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}\n{'='*70}")

    os.makedirs(args.resultPath, exist_ok=True)
    with open(f'{args.resultPath}/summary_{args.dataType}_{density:.2f}_runs{num_runs}.txt', 'w') as f:
        f.write(f"Density: {density}\nRuns: {num_runs}\nEpochs: {args.epochNum}\n")
        f.write(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}\nRMSE: {avg_rmse:.4f} ± {std_rmse:.4f}\n")
        for i, (mae, rmse) in enumerate(zip(all_mae, all_rmse)):
            f.write(f"Run {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}\n")
    return avg_mae, avg_rmse


def main():
    parser = argparse.ArgumentParser(description="LDCF for RT")
    parser.add_argument('--dataType', default='rt', type=str, help='rt or tp')
    parser.add_argument('--density', default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], type=list, help='Matrix densities')
    parser.add_argument('--epochNum', default=50, type=int)
    parser.add_argument('--num_runs', default=20, type=int)
    parser.add_argument('--batchSize', default=256, type=int)
    parser.add_argument('--layers', default=[64, 32, 16, 8, 1], type=list)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--verbose', default=10, type=int)
    parser.add_argument('--store', default=True, type=bool)
    parser.add_argument('--modelPath', default='./Model', type=str)
    parser.add_argument('--resultPath', default='./Result_20260410', type=str)
    args = parser.parse_args()

    if isinstance(args.density, (int, float)):
        args.density = [args.density]

    print(f"PyTorch {torch.__version__}, Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    os.makedirs(args.modelPath, exist_ok=True)
    os.makedirs(args.resultPath, exist_ok=True)

    for density in args.density:
        train_multiple_runs(args, density, args.num_runs)


if __name__ == '__main__':
    main()
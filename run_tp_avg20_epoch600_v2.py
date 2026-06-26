"""
Created on 2018/10/21 by Chunhui Yin.
Reproduced on 2026/4/2 by Nan Sun.
Optimized PyTorch version for TP with:
- log1p transform for TP values
- Adaptive delta for Huber loss
- Learning rate scheduling + Early stopping
- Dropout for regularization
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    def __init__(self, num_users, num_services, layers, dropout_rate=0.2):
        super(LDCFModel, self).__init__()
        self.num_users = num_users
        self.num_services = num_services
        self.layers = layers
        self.emb_dim = layers[0] // 4

        # Embeddings
        self.user_id_embed = nn.Embedding(num_users, self.emb_dim)
        self.user_loc_embed = nn.Embedding(num_users, self.emb_dim)
        self.service_id_embed = nn.Embedding(num_services, self.emb_dim)
        self.service_loc_embed = nn.Embedding(num_services, self.emb_dim)

        # MLP
        mlp_input_dim = self.emb_dim * 6  # user_id + user_loc(2*emb_dim) + service_id + service_loc(2*emb_dim)
        mlp_layers = []
        current_dim = mlp_input_dim
        for i in range(1, len(layers) - 1):
            mlp_layers.append(nn.Linear(current_dim, layers[i]))
            mlp_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))
            current_dim = layers[i]
        self.mlp = nn.Sequential(*mlp_layers)

        # Final layer (MLP output + cosine similarity)
        self.final_layer = nn.Linear(layers[-2] + 1, layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_id, user_loc, service_id, service_loc):
        user_id_vec = self.user_id_embed(user_id)
        user_loc_vec = self.user_loc_embed(user_loc).view(user_loc.size(0), -1)
        service_id_vec = self.service_id_embed(service_id)
        service_loc_vec = self.service_loc_embed(service_loc).view(service_loc.size(0), -1)

        mlp_input = torch.cat([user_id_vec, user_loc_vec, service_id_vec, service_loc_vec], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Cosine similarity (Adaptive Corrector)
        user_loc_emb = self.user_loc_embed(user_loc)
        service_loc_emb = self.service_loc_embed(service_loc)
        user_loc_avg = user_loc_emb.mean(dim=1)
        service_loc_avg = service_loc_emb.mean(dim=1)
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
        self.patience = args.patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LDCFModel(self.num_users, self.num_services, args.layers,
                               dropout_rate=args.dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        """
        10轮不降就早停
        """

        # For early stopping
        self.best_mae = float('inf')
        self.early_stop_counter = 0

        # Dynamic delta for Huber loss
        self.delta = args.delta_init

    def run(self):
        # Initial evaluation
        mae, rmse = evaluate_loader(self.model, self.test_loader)
        self.best_mae = mae
        best_rmse = rmse
        best_epoch = -1
        eval_results = []

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

                self.optimizer.zero_grad()
                pred = self.model(user_id, user_geo, service_id, service_geo)
                loss = huber_loss(pred.squeeze(), qos, delta=self.delta)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(user_id)

            avg_loss = total_loss / len(self.train_loader.dataset)

            # Adjust delta dynamically based on current MAE (only for TP)
            if self.args.dataType == 'tp' and epoch % 20 == 0 and epoch > 0:
                new_delta = max(5, self.best_mae * 1.2)   # 下限降到5，系数降到1.2
                """
                动态 delta 调整
                """
                if new_delta != self.delta:
                    self.delta = new_delta
                    print(f"  -> Updating Huber delta to {self.delta:.2f}")

            if epoch % self.verbose == 0:
                mae, rmse = evaluate_loader(self.model, self.test_loader)
                eval_results.append([mae, rmse])
                self.scheduler.step(mae)

                if mae < self.best_mae:
                    self.best_mae = mae
                    best_rmse = rmse
                    best_epoch = epoch
                    self.early_stop_counter = 0
                    if self.store:
                        self.saveModel()
                else:
                    self.early_stop_counter += 1

                print(f'Epoch {epoch:3d}: MAE = {mae:.4f} | RMSE = {rmse:.4f} | Loss = {avg_loss:.4f} | Delta = {self.delta:.2f}')

                # Early stopping
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print(f'Best at epoch {best_epoch}: MAE = {self.best_mae:.4f} | RMSE = {best_rmse:.4f}')
        if self.store and eval_results:
            result_path_with_run = f'{self.resultPath}/run_{self.run_id}'
            saveResult(result_path_with_run, self.dataset.dataType, self.dataset.density,
                       np.array(eval_results), ['MAE', 'RMSE'])
        return self.best_mae, best_rmse

    def saveModel(self):
        os.makedirs(self.modelPath, exist_ok=True)
        torch.save(self.model.state_dict(),
                   f'{self.modelPath}/{self.dataset.dataType}_{self.dataset.density:.2f}_{self.args.layers}_run{self.run_id}.pth')


def train_multiple_runs(args, density, num_runs=20):
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
    parser = argparse.ArgumentParser(description="Optimized LDCF for TP")
    parser.add_argument('--dataType', default='tp', type=str, help='rt or tp')
    parser.add_argument('--density', default=[0.05], type=list, help='Matrix densities (single value for quick test)')
    parser.add_argument('--epochNum', default=400, type=int)
    parser.add_argument('--num_runs', default=10, type=int)      # 测试时先用1，效果好再加大
    parser.add_argument('--batchSize', default=512, type=int)   # 增大batch_size稳定梯度
    parser.add_argument('--layers', default=[128, 64, 32, 16, 1], type=list)  # 增大embedding能力
    """
     [64,32,16,8,1]	→ [128,64,32,16,1]
    """
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    """
    Dropout防止过拟合
    """
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-5, type=float, help='Weight decay (L2 regularization)')
    """
    weight_decay=1e-5
    """
    parser.add_argument('--verbose', default=10, type=int)
    parser.add_argument('--store', default=True, type=bool)
    parser.add_argument('--modelPath', default='./Model', type=str)
    parser.add_argument('--resultPath', default='./Result_20260415', type=str)
    parser.add_argument('--patience', default=30, type=int, help='Early stopping patience')
    parser.add_argument('--delta_init', default=50.0, type=float, help='Initial delta for Huber loss')
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
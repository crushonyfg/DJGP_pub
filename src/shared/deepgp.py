# python DeepGP_test.py --folder_name "2025_04_01_23" --num_epochs 200 --patience 5 --batch_size 1024 --lr 0.01
import os
import pickle
import argparse
import torch
import time
import gpytorch
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', device="cuda"):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims, dtype=torch.float32, device=device)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims, dtype=torch.float32, device=device)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGP(DeepGP):
    def __init__(self, train_x_shape, output_dims):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=output_dims,  # num_hidden_dims
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

# def train_model(model, train_loader, test_loader, optimizer, mll, args):
#     best_loss = float('inf')
#     patience_counter = 0
#     final_metrics = None
#     num_samples = 3

#     start_time = time.time()
#     epochs_iter = tqdm.tqdm(range(args.num_epochs), desc="Epoch")
#     for epoch in epochs_iter:
#         # 训练模式
#         model.train()
#         epoch_loss = 0.0
#         num_batches = 0
        
#         # 对每个 minibatch 进行训练
#         minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
#         for x_batch, y_batch in minibatch_iter:
#             with gpytorch.settings.num_likelihood_samples(num_samples):
#                 optimizer.zero_grad()
#                 output = model(x_batch)
#                 loss = -mll(output, y_batch)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#                 num_batches += 1
#                 minibatch_iter.set_postfix(loss=loss.item())
        
#         # 计算平均损失
#         avg_epoch_loss = epoch_loss / num_batches
        
#         # 早停检查
#         if avg_epoch_loss < best_loss:
#             best_loss = avg_epoch_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
        
#         # 模型评估
#         model.eval()
#         with torch.no_grad():
#             predictive_means, predictive_variances, test_lls = model.predict(test_loader)
        
#         # 计算评估指标
#         rmse = torch.mean((predictive_means.mean(0) - test_loader.dataset.tensors[1])**2).sqrt().item()
#         nlpd_values = -test_lls
#         nlpd_25 = torch.quantile(nlpd_values, 0.25).item()
#         nlpd_50 = torch.quantile(nlpd_values, 0.50).item()
#         nlpd_75 = torch.quantile(nlpd_values, 0.75).item()
        
#         # 保存最后一轮的指标
#         final_metrics = {
#             'rmse': rmse,
#             'nlpd_25': nlpd_25,
#             'nlpd_50': nlpd_50,
#             'nlpd_75': nlpd_75,
#             'run_time': time.time() - start_time
#         }
        
#         # 更新进度条信息
#         epochs_iter.set_postfix(
#             loss=f"{avg_epoch_loss:.4f}",
#             rmse=f"{rmse:.4f}",
#             nlpd_50=f"{nlpd_50:.4f}"
#         )
        
#         # 检查是否需要早停
#         if patience_counter >= args.patience:
#             print(f"\nEarly stopping triggered after {epoch + 1} epochs")
#             break
    
#     return final_metrics

import time, tqdm, math, torch

def train_model(model, train_loader, test_loader, optimizer, mll, args):
    best_loss = float('inf')
    patience_counter = 0
    final_metrics = None
    num_samples = 3

    start_time = time.time()
    epochs_iter = tqdm.tqdm(range(args.num_epochs), desc="Epoch")
    for epoch in epochs_iter:
        # ——— 训练环节 ———
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in tqdm.tqdm(train_loader, desc="Minibatch", leave=False):
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        # 早停逻辑
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # ——— 评估环节 ———
        model.eval()
        with torch.no_grad():
            predictive_means, predictive_variances, _ = model.predict(test_loader)
        
        # 提取参数
        mu = predictive_means.mean(0)                # [N_test]
        sigma = predictive_variances.sqrt()          # [N_test]
        y_true = test_loader.dataset.tensors[1]      # [N_test]
        
        # RMSE
        rmse = torch.sqrt(torch.mean((mu - y_true)**2)).item()
        
        # CRPS
        z = (y_true - mu) / sigma
        cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        pdf = torch.exp(-0.5 * z**2) / math.sqrt(2*math.pi)
        crps_vals = sigma * ( z * (2*cdf - 1) + 2*pdf - 1/math.sqrt(math.pi) )
        mean_crps = crps_vals.mean().item()
        
        # 保存
        final_metrics = {
            'rmse':   rmse,
            'crps':   mean_crps,
            'run_time': time.time() - start_time
        }
        
        # 更新进度条
        epochs_iter.set_postfix(
            loss=f"{avg_epoch_loss:.4f}",
            rmse=f"{rmse:.4f}",
            crps=f"{mean_crps:.4f}"
        )
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    return final_metrics


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name containing dataset.pkl')
    parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=2, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    dataset_path = os.path.join(args.folder_name, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_train = dataset["X_train"].float().to(device)
    Y_train = dataset["Y_train"].float().to(device)
    X_test = dataset["X_test"].float().to(device)
    Y_test = dataset["Y_test"].float().to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 初始化模型
    model = DeepGP(X_train.shape, args.hidden_dim).to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))

    # 训练模型
    final_metrics = train_model(model, train_loader, test_loader, optimizer, mll, args)

    # 打印最终结果
    print("\nFinal metrics:")
    print(f"RMSE: {final_metrics['rmse']:.4f}")
    print(f"CRPS: {final_metrics['crps']:.4f}")

    # # 保存结果
    # results = {
    #     'final_metrics': final_metrics,
    #     'args': vars(args)
    # }
    
    # # 创建结果文件夹（如果不存在）
    # os.makedirs('results', exist_ok=True)
    # results_path = os.path.join('results', f'results_{args.folder_name}.pkl')
    # with open(results_path, 'wb') as f:
    #     pickle.dump(results, f)
    # print(f"\nResults saved to {results_path}")
    # return [final_metrics['rmse'], final_metrics['nlpd_25'], final_metrics['nlpd_50'], final_metrics['nlpd_75'], final_metrics['run_time']]
    return [final_metrics['rmse'], final_metrics['crps'], final_metrics['run_time']]

if __name__ == "__main__":
    main()

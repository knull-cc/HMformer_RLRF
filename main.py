from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
# ===== 修改开始：导入独立多目标损失模块，训练循环仍保持 criterion(pred, true) 接口 =====
from utils.custom_losses import MultiObjectiveLoss
# ===== 修改结束：导入独立多目标损失模块，训练循环仍保持 criterion(pred, true) 接口 =====
from tqdm import tqdm
from models.HMformer import HMformer

# ===== 修改开始：删除非 HMformer 模型导入，当前入口只保留 HMformer =====
# 当前最简 demo 只运行 HMformer，不再导入其他模型。
# ===== 修改结束：删除非 HMformer 模型导入，当前入口只保留 HMformer =====


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ===== 修改开始：入口描述改为 HMformer，避免保留其他模型名称造成误解 =====
parser = argparse.ArgumentParser(description='HMformer')
# ===== 修改结束：入口描述改为 HMformer，避免保留其他模型名称造成误解 =====

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

# ===== 修改开始：新增 loss_mode 和静态加权参数，baseline 分支继续使用原始 loss_func =====
parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--loss_mode', type=str, default='baseline',
                    choices=['baseline', 'point', 'point_direction', 'point_direction_trend'])
parser.add_argument('--lambda_p', type=float, default=1.0)
parser.add_argument('--lambda_d', type=float, default=1.0)
parser.add_argument('--lambda_t', type=float, default=1.0)
# ===== 修改结束：新增 loss_mode 和静态加权参数，baseline 分支继续使用原始 loss_func =====
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
# ===== 修改开始：只允许选择 HMformer，避免误传其他模型名进入无效分支 =====
parser.add_argument('--model', type=str, default='HMformer', choices=['HMformer'])
# ===== 修改结束：只允许选择 HMformer，避免误传其他模型名进入无效分支 =====
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--ifatten', type=int, default=0)
parser.add_argument('--fusion', type=int, default=2)
# ===== 修改开始：删除重复 --stride 注册，避免 argparse 因重复参数直接报错 =====
# 原代码在这里第二次注册 --stride；保留上方唯一的 --stride 参数定义。
# ===== 修改结束：删除重复 --stride 注册，避免 argparse 因重复参数直接报错 =====


args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    # ===== 修改开始：自动选择 GPU 或 CPU，避免无 CUDA 环境下最小 demo 无法启动 =====
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ===== 修改结束：自动选择 GPU 或 CPU，避免无 CUDA 环境下最小 demo 无法启动 =====

    time_now = time.time()
    train_steps = len(train_loader)
    # ===== 修改开始：删除其他模型构造分支，当前训练入口只实例化 HMformer =====
    model = HMformer(args, device)
    model.to(device)
    # ===== 修改结束：删除其他模型构造分支，当前训练入口只实例化 HMformer =====
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # ===== 修改开始：按 loss_mode 切换 baseline 原始损失或最简多目标静态加权损失 =====
    if args.loss_mode == 'baseline':
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        else:
            raise ValueError('baseline loss_func must be mse or smape, got {}'.format(args.loss_func))
    else:
        criterion = MultiObjectiveLoss(
            loss_mode=args.loss_mode,
            lambda_p=args.lambda_p,
            lambda_d=args.lambda_d,
            lambda_t=args.lambda_t
        )
    # ===== 修改结束：按 loss_mode 切换 baseline 原始损失或最简多目标静态加权损失 =====
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            outputs = model(batch_x, ii)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)

mses = np.array(mses)
maes = np.array(maes)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

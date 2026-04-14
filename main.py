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
# ===== 修改开始：freq 改为字符串，支持 ETTm1 使用 t 分钟频率，同时兼容旧脚本 0 表示小时级 =====
parser.add_argument('--freq', type=str, default='h')
# ===== 修改结束：freq 改为字符串，支持 ETTm1 使用 t 分钟频率，同时兼容旧脚本 0 表示小时级 =====
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
# ===== 修改开始：新增训练日志打印间隔，避免每 1000 iter 才显示一次 loss =====
parser.add_argument('--log_interval', type=int, default=50)
# ===== 修改结束：新增训练日志打印间隔，避免每 1000 iter 才显示一次 loss =====
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
parser.add_argument('--loss_mode', type=str, default='point',
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

# ===== 修改开始：保护 log_interval，避免用户传 0 或负数导致取模报错 =====
if args.log_interval <= 0:
    args.log_interval = 1
# ===== 修改结束：保护 log_interval，避免用户传 0 或负数导致取模报错 =====

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

# ===== 修改开始：新增 loss 日志格式化函数，用于展示 baseline 或多目标 loss 分项 =====
def format_loss_log(loss, criterion, args):
    if args.loss_mode == 'baseline':
        return 'baseline_loss: {:.7f}'.format(loss.item())

    last_losses = getattr(criterion, 'last_losses', {})
    log_parts = ['total: {:.7f}'.format(loss.item())]
    for loss_name in ['point', 'direction', 'trend']:
        loss_value = last_losses.get(loss_name)
        if loss_value is not None:
            log_parts.append('{}: {:.7f}'.format(loss_name, loss_value.item()))
    return ' | '.join(log_parts)
# ===== 修改结束：新增 loss 日志格式化函数，用于展示 baseline 或多目标 loss 分项 =====

mses = []
maes = []

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    # ===== 修改开始：兼容旧脚本传入 --freq 0，同时允许 ETTm1 传入 --freq t =====
    if str(args.freq) == '0':
        args.freq = 'h'
    # ===== 修改结束：兼容旧脚本传入 --freq 0，同时允许 ETTm1 传入 --freq t =====

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # ===== 修改开始：只对 tsf_data 应用季节周期映射，避免 ETTm1 的 t 频率被误查 SEASONALITY_MAP =====
    if args.data == 'tsf_data' and args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    # ===== 修改结束：只对 tsf_data 应用季节周期映射，避免 ETTm1 的 t 频率被误查 SEASONALITY_MAP =====

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

    # ===== 修改开始：打印当前实验配置，明确展示模型、设备、数据长度和 loss 选择 =====
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print("========== HMformer 实验配置 ==========")
    print("setting: {}".format(setting))
    print("device: {}".format(device_name))
    print("model: {}".format(args.model))
    print("data: {}, root_path: {}, data_path: {}".format(args.data, args.root_path, args.data_path))
    print("seq_len: {}, label_len: {}, pred_len: {}".format(args.seq_len, args.label_len, args.pred_len))
    print("batch_size: {}, train_epochs: {}, train_steps_per_epoch: {}".format(args.batch_size, args.train_epochs, train_steps))
    print("loss_mode: {}".format(args.loss_mode))
    if args.loss_mode == 'baseline':
        print("baseline loss_func: {}".format(args.loss_func))
    else:
        print("loss weights -> lambda_p: {}, lambda_d: {}, lambda_t: {}".format(args.lambda_p, args.lambda_d, args.lambda_t))
    print("log_interval: {}".format(args.log_interval))
    print("======================================")
    # ===== 修改结束：打印当前实验配置，明确展示模型、设备、数据长度和 loss 选择 =====

    # ===== 修改开始：记录是否已经打印首个 batch shape，用于确认多变量输入是否生效 =====
    shape_logged = False
    # ===== 修改结束：记录是否已经打印首个 batch shape，用于确认多变量输入是否生效 =====

    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        # ===== 修改开始：给 tqdm 增加 total 和 desc，让训练进度显示 x/y 与百分比 =====
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader),
            total=train_steps,
            desc='train epoch {}/{}'.format(epoch + 1, args.train_epochs)
        ):
        # ===== 修改结束：给 tqdm 增加 total 和 desc，让训练进度显示 x/y 与百分比 =====

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # ===== 修改开始：保留裁剪前 batch_y shape，便于日志验证真实标签窗口长度 =====
            raw_batch_y_shape = tuple(batch_y.shape)
            # ===== 修改结束：保留裁剪前 batch_y shape，便于日志验证真实标签窗口长度 =====
            outputs = model(batch_x, ii)
            # ===== 修改开始：保留裁剪前 outputs shape，便于日志验证模型原始输出维度 =====
            raw_outputs_shape = tuple(outputs.shape)
            # ===== 修改结束：保留裁剪前 outputs shape，便于日志验证模型原始输出维度 =====

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            # ===== 修改开始：首个 batch 打印输入输出 shape，确认是否为 [B, seq_len, enc_in] 多变量形式 =====
            if not shape_logged:
                print("shape check -> batch_x: {}, raw_batch_y: {}, raw_outputs: {}, cropped_outputs: {}, cropped_batch_y: {}".format(
                    tuple(batch_x.shape),
                    raw_batch_y_shape,
                    raw_outputs_shape,
                    tuple(outputs.shape),
                    tuple(batch_y.shape)
                ))
                shape_logged = True
            # ===== 修改结束：首个 batch 打印输入输出 shape，确认是否为 [B, seq_len, enc_in] 多变量形式 =====
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            # ===== 修改开始：按 log_interval 打印 total/point/direction/trend，最后一个 step 也会打印 =====
            should_log = (i + 1) % args.log_interval == 0 or (i + 1) == train_steps
            if should_log:
                tqdm.write("\titers: {0}/{1}, epoch: {2} | {3}".format(
                    i + 1,
                    train_steps,
                    epoch + 1,
                    format_loss_log(loss, criterion, args)
                ))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            # ===== 修改结束：按 log_interval 打印 total/point/direction/trend，最后一个 step 也会打印 =====
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

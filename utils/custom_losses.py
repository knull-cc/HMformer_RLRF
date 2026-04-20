import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 修改开始：扩展完整 feedback 多目标损失模块，支持 point、direction、trend、vol、bias、lag 六类静态加权损失 =====
class MultiObjectiveLoss(nn.Module):
    """
    完整反馈多目标损失：
    - point：逐点 MSE，保证数值准确
    - direction：一阶差分方向 BCE，保证涨跌方向
    - trend：一阶差分 MSE，保证趋势形状
    - vol：一阶差分波动强度 MSE，约束波动强弱
    - bias：预测残差均值平方，约束整体偏高或偏低
    - lag：可导 soft lag，约束预测变化不要整体提前或滞后
    """

    def __init__(self, loss_mode, lambda_p=1.0, lambda_d=0.1, lambda_t=0.5,
                 lambda_v=0.1, lambda_b=0.1, lambda_lag=0.05,
                 lag_k=3, lag_tau=0.1):
        super(MultiObjectiveLoss, self).__init__()
        valid_modes = ['feedback']
        if loss_mode not in valid_modes:
            raise ValueError('loss_mode must be one of {}'.format(valid_modes))

        self.loss_mode = loss_mode
        self.lambda_p = lambda_p
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.lambda_v = lambda_v
        self.lambda_b = lambda_b
        self.lambda_lag = lambda_lag
        self.lag_k = max(int(lag_k), 0)
        self.lag_tau = max(float(lag_tau), 1e-6)
        self.eps = 1e-6
        self.point_loss = nn.MSELoss()
        self.trend_loss = nn.MSELoss()
        self.vol_loss = nn.MSELoss()
        # ===== 修改开始：记录最近一次 loss 分项，便于训练日志展示完整 feedback loss =====
        self.last_losses = {}
        # ===== 修改结束：记录最近一次 loss 分项，便于训练日志展示完整 feedback loss =====

    def _lag_loss(self, pred_diff, true_diff):
        # ===== 修改开始：使用 soft lag 估计预测变化与真实变化的时间错位，避免不可导 argmax =====
        max_lag = min(self.lag_k, pred_diff.shape[1] - 1)
        if max_lag <= 0:
            return pred_diff.new_tensor(0.0)

        lag_errors = []
        lag_values = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shift = -lag
                pred_slice = pred_diff[:, :-shift, :]
                true_slice = true_diff[:, shift:, :]
            elif lag > 0:
                pred_slice = pred_diff[:, lag:, :]
                true_slice = true_diff[:, :-lag, :]
            else:
                pred_slice = pred_diff
                true_slice = true_diff

            lag_error = torch.mean((pred_slice - true_slice) ** 2, dim=1)
            lag_errors.append(lag_error)
            lag_values.append(lag)

        lag_errors = torch.stack(lag_errors, dim=-1)
        lag_tensor = pred_diff.new_tensor(lag_values).view(1, 1, -1)
        lag_weights = torch.softmax(-lag_errors / self.lag_tau, dim=-1)
        expected_abs_lag = torch.sum(lag_weights * torch.abs(lag_tensor), dim=-1) / max_lag
        return torch.mean(expected_abs_lag)
        # ===== 修改结束：使用 soft lag 估计预测变化与真实变化的时间错位，避免不可导 argmax =====

    def forward(self, pred, true):
        loss_point = self.point_loss(pred, true)
        total_loss = self.lambda_p * loss_point
        # ===== 修改开始：feedback 模式一次性计算六类可导反馈项，并用静态权重求和 =====
        loss_direction = pred.new_tensor(0.0)
        loss_trend = pred.new_tensor(0.0)
        loss_vol = pred.new_tensor(0.0)
        loss_bias = torch.mean(torch.mean(pred - true, dim=1) ** 2)
        loss_lag = pred.new_tensor(0.0)

        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            true_diff = true[:, 1:, :] - true[:, :-1, :]

            direction_target = (true_diff > 0).float()
            loss_direction = F.binary_cross_entropy_with_logits(pred_diff, direction_target)

            loss_trend = self.trend_loss(pred_diff, true_diff)

            pred_vol = torch.sqrt(torch.var(pred_diff, dim=1, unbiased=False) + self.eps)
            true_vol = torch.sqrt(torch.var(true_diff, dim=1, unbiased=False) + self.eps)
            loss_vol = self.vol_loss(pred_vol, true_vol)

            loss_lag = self._lag_loss(pred_diff, true_diff)

        total_loss = (
            total_loss
            + self.lambda_d * loss_direction
            + self.lambda_t * loss_trend
            + self.lambda_v * loss_vol
            + self.lambda_b * loss_bias
            + self.lambda_lag * loss_lag
        )
        # ===== 修改结束：feedback 模式一次性计算六类可导反馈项，并用静态权重求和 =====

        # ===== 修改开始：保存 detached loss 分项供日志读取，不改变返回的 total_loss =====
        self.last_losses = {
            'total': total_loss.detach(),
            'point': loss_point.detach(),
            'direction': loss_direction.detach(),
            'trend': loss_trend.detach(),
            'vol': loss_vol.detach(),
            'bias': loss_bias.detach(),
            'lag': loss_lag.detach()
        }
        # ===== 修改结束：保存 detached loss 分项供日志读取，不改变返回的 total_loss =====
        return total_loss
# ===== 修改结束：扩展完整 feedback 多目标损失模块，支持 point、direction、trend、vol、bias、lag 六类静态加权损失 =====

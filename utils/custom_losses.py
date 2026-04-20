import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 修改开始：扩展完整 feedback 多目标损失模块，支持静态加权和 batch EMA 自适应加权 =====
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
                 lag_k=3, lag_tau=0.1,
                 weight_mode='static', ema_beta=0.9, weight_tau=1.0):
        super(MultiObjectiveLoss, self).__init__()
        valid_modes = ['feedback']
        if loss_mode not in valid_modes:
            raise ValueError('loss_mode must be one of {}'.format(valid_modes))
        # ===== 修改开始：新增静态/动态加权模式校验，dynamic_ema 基于 batch loss 严重度调整权重 =====
        valid_weight_modes = ['static', 'dynamic_ema']
        if weight_mode not in valid_weight_modes:
            raise ValueError('weight_mode must be one of {}'.format(valid_weight_modes))
        # ===== 修改结束：新增静态/动态加权模式校验，dynamic_ema 基于 batch loss 严重度调整权重 =====

        self.loss_mode = loss_mode
        self.lambda_p = lambda_p
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.lambda_v = lambda_v
        self.lambda_b = lambda_b
        self.lambda_lag = lambda_lag
        self.lag_k = max(int(lag_k), 0)
        self.lag_tau = max(float(lag_tau), 1e-6)
        # ===== 修改开始：保存 batch EMA 动态加权配置和每个 loss 分项的历史均值 =====
        self.weight_mode = weight_mode
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.999)
        self.weight_tau = max(float(weight_tau), 1e-6)
        self.loss_ema = {}
        # ===== 修改结束：保存 batch EMA 动态加权配置和每个 loss 分项的历史均值 =====
        self.eps = 1e-6
        self.point_loss = nn.MSELoss()
        self.trend_loss = nn.MSELoss()
        self.vol_loss = nn.MSELoss()
        # ===== 修改开始：记录最近一次 loss 分项，便于训练日志展示完整 feedback loss =====
        self.last_losses = {}
        # ===== 修改结束：记录最近一次 loss 分项，便于训练日志展示完整 feedback loss =====
        # ===== 修改开始：记录最近一次有效权重，便于观察 dynamic_ema 是否按 batch 状态调整 =====
        self.last_weights = {}
        # ===== 修改结束：记录最近一次有效权重，便于观察 dynamic_ema 是否按 batch 状态调整 =====

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

    def _build_weights(self, loss_values, base_lambdas):
        # ===== 修改开始：根据 weight_mode 生成当前 batch 的有效 loss 权重 =====
        effective_weights = {}
        for loss_name, base_lambda in base_lambdas.items():
            effective_weights[loss_name] = loss_values[loss_name].new_tensor(float(base_lambda))

        if self.weight_mode == 'static':
            return effective_weights

        active_names = [
            loss_name for loss_name, base_lambda in base_lambdas.items()
            if float(base_lambda) > 0.0
        ]
        if len(active_names) == 0:
            return effective_weights

        severity_values = []
        for loss_name in active_names:
            current_loss = loss_values[loss_name].detach()
            if loss_name not in self.loss_ema or self.loss_ema[loss_name].device != current_loss.device:
                self.loss_ema[loss_name] = current_loss
            else:
                self.loss_ema[loss_name] = (
                    self.ema_beta * self.loss_ema[loss_name]
                    + (1.0 - self.ema_beta) * current_loss
                ).detach()

            severity_values.append(current_loss / (self.loss_ema[loss_name] + self.eps))

        severities = torch.stack(severity_values)
        dynamic_probs = torch.softmax(severities / self.weight_tau, dim=0)
        dynamic_multipliers = dynamic_probs * len(active_names)

        for idx, loss_name in enumerate(active_names):
            base_weight = loss_values[loss_name].new_tensor(float(base_lambdas[loss_name]))
            effective_weights[loss_name] = base_weight * dynamic_multipliers[idx]

        return effective_weights
        # ===== 修改结束：根据 weight_mode 生成当前 batch 的有效 loss 权重 =====

    def forward(self, pred, true):
        loss_point = self.point_loss(pred, true)
        # ===== 修改开始：feedback 模式一次性计算六类可导反馈项，并用静态或动态权重求和 =====
        loss_direction = pred.new_tensor(0.0)
        loss_trend = pred.new_tensor(0.0)
        loss_vol = pred.new_tensor(0.0)
        loss_bias = pred.new_tensor(0.0)
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

            if self.lambda_lag > 0:
                loss_lag = self._lag_loss(pred_diff, true_diff)

        if self.lambda_b > 0:
            loss_bias = torch.mean(torch.mean(pred - true, dim=1) ** 2)

        loss_values = {
            'point': loss_point,
            'direction': loss_direction,
            'trend': loss_trend,
            'vol': loss_vol,
            'bias': loss_bias,
            'lag': loss_lag
        }
        base_lambdas = {
            'point': self.lambda_p,
            'direction': self.lambda_d,
            'trend': self.lambda_t,
            'vol': self.lambda_v,
            'bias': self.lambda_b,
            'lag': self.lambda_lag
        }
        effective_weights = self._build_weights(loss_values, base_lambdas)

        total_loss = (
            effective_weights['point'] * loss_point
            + effective_weights['direction'] * loss_direction
            + effective_weights['trend'] * loss_trend
            + effective_weights['vol'] * loss_vol
            + effective_weights['bias'] * loss_bias
            + effective_weights['lag'] * loss_lag
        )
        # ===== 修改结束：feedback 模式一次性计算六类可导反馈项，并用静态或动态权重求和 =====

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
        # ===== 修改开始：保存 detached 有效权重供日志读取，不参与反向传播 =====
        self.last_weights = {
            loss_name: loss_weight.detach()
            for loss_name, loss_weight in effective_weights.items()
        }
        # ===== 修改结束：保存 detached 有效权重供日志读取，不参与反向传播 =====
        return total_loss
# ===== 修改结束：扩展完整 feedback 多目标损失模块，支持静态加权和 batch EMA 自适应加权 =====

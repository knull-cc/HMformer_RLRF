import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 修改开始：新增最简多目标损失模块，支持 point、direction、trend 三类静态加权损失 =====
class MultiObjectiveLoss(nn.Module):
    """
    最简多目标损失：
    - point：逐点 MSE，保证数值准确
    - direction：一阶差分方向 BCE，保证涨跌方向
    - trend：一阶差分 MSE，保证趋势形状
    """

    def __init__(self, loss_mode, lambda_p=1.0, lambda_d=1.0, lambda_t=1.0):
        super(MultiObjectiveLoss, self).__init__()
        valid_modes = ['point', 'point_direction', 'point_direction_trend']
        if loss_mode not in valid_modes:
            raise ValueError('loss_mode must be one of {}'.format(valid_modes))

        self.loss_mode = loss_mode
        self.lambda_p = lambda_p
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.point_loss = nn.MSELoss()
        self.trend_loss = nn.MSELoss()
        # ===== 修改开始：记录最近一次 loss 分项，便于训练日志展示 point、direction、trend =====
        self.last_losses = {}
        # ===== 修改结束：记录最近一次 loss 分项，便于训练日志展示 point、direction、trend =====

    def forward(self, pred, true):
        loss_point = self.point_loss(pred, true)
        total_loss = self.lambda_p * loss_point
        # ===== 修改开始：初始化当前 batch 的 loss 分项记录，不影响 loss 反向传播 =====
        loss_direction = None
        loss_trend = None
        # ===== 修改结束：初始化当前 batch 的 loss 分项记录，不影响 loss 反向传播 =====

        if self.loss_mode in ['point_direction', 'point_direction_trend']:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            true_diff = true[:, 1:, :] - true[:, :-1, :]

            direction_target = (true_diff > 0).float()
            loss_direction = F.binary_cross_entropy_with_logits(pred_diff, direction_target)
            total_loss = total_loss + self.lambda_d * loss_direction

            if self.loss_mode == 'point_direction_trend':
                loss_trend = self.trend_loss(pred_diff, true_diff)
                total_loss = total_loss + self.lambda_t * loss_trend

        # ===== 修改开始：保存 detached loss 分项供日志读取，不改变返回的 total_loss =====
        self.last_losses = {
            'total': total_loss.detach(),
            'point': loss_point.detach(),
            'direction': None if loss_direction is None else loss_direction.detach(),
            'trend': None if loss_trend is None else loss_trend.detach()
        }
        # ===== 修改结束：保存 detached loss 分项供日志读取，不改变返回的 total_loss =====
        return total_loss
# ===== 修改结束：新增最简多目标损失模块，支持 point、direction、trend 三类静态加权损失 =====

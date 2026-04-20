# ===== 修改开始：新增 Exchange 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，删除可切换模型名变量 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，删除可切换模型名变量 =====
# ===== 修改开始：新增 feedback 完整损失的 vol、bias、lag 权重和 soft lag 配置 =====
lambda_p=${LAMBDA_P:-1.0}
lambda_d=${LAMBDA_D:-0.1}
lambda_t=${LAMBDA_T:-0.5}
lambda_v=${LAMBDA_V:-0.1}
lambda_b=${LAMBDA_B:-0.1}
lambda_lag=${LAMBDA_LAG:-0.05}
lag_k=${LAG_K:-3}
lag_tau=${LAG_TAU:-0.1}
# ===== 修改结束：新增 feedback 完整损失的 vol、bias、lag 权重和 soft lag 配置 =====
# ===== 修改开始：完整测试默认使用更充分的 epoch、更大的 batch 和更多数据加载进程 =====
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-256}
num_workers=${NUM_WORKERS:-8}
# ===== 修改结束：完整测试默认使用更充分的 epoch、更大的 batch 和更多数据加载进程 =====
log_interval=${LOG_INTERVAL:-50}
# ===== 修改开始：Exchange 720 horizon 需要完整训练集窗口，避免 percent=10 导致 __len__ 为负 =====
percent=${PERCENT:-100}
# ===== 修改结束：Exchange 720 horizon 需要完整训练集窗口，避免 percent=10 导致 __len__ 为负 =====

# ===== 修改开始：打印脚本级实验配置，明确当前选择的 loss 和训练参数 =====
echo "========== HMformer Exchange 脚本配置 =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOSS_MODE=${loss_mode}"
# ===== 修改开始：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "LAMBDA_P=${lambda_p}, LAMBDA_D=${lambda_d}, LAMBDA_T=${lambda_t}"
echo "LAMBDA_V=${lambda_v}, LAMBDA_B=${lambda_b}, LAMBDA_LAG=${lambda_lag}"
echo "LAG_K=${lag_k}, LAG_TAU=${lag_tau}"
# ===== 修改结束：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "TRAIN_EPOCHS=${train_epochs}, BATCH_SIZE=${batch_size}, NUM_WORKERS=${num_workers}"
echo "PERCENT=${percent}, LOG_INTERVAL=${log_interval}"
echo "pred_len list: 96 192 336 720"
echo "==============================================="
# ===== 修改结束：打印脚本级实验配置，明确当前选择的 loss 和训练参数 =====

run_exchange() {
  pred_len=$1

  # ===== 修改开始：每个 horizon 开始前打印当前运行任务，便于对应日志和结果 =====
  echo "---------- start Exchange_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 开始前打印当前运行任务，便于对应日志和结果 =====

  # ===== 修改开始：调用 main.py 时传入 feedback 完整损失参数 =====
  python -u main.py \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_96_${pred_len}_${loss_mode} \
    --model HMformer \
    --data custom \
    --features M \
    --freq 0 \
    --percent ${percent} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --enc_in 8 \
    --c_out 8 \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --train_epochs ${train_epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --log_interval ${log_interval} \
    --loss_mode ${loss_mode} \
    --lambda_p ${lambda_p} \
    --lambda_d ${lambda_d} \
    --lambda_t ${lambda_t} \
    --lambda_v ${lambda_v} \
    --lambda_b ${lambda_b} \
    --lambda_lag ${lambda_lag} \
    --lag_k ${lag_k} \
    --lag_tau ${lag_tau}

  # ===== 修改结束：调用 main.py 时传入 feedback 完整损失参数 =====

  # ===== 修改开始：每个 horizon 结束后打印完成提示，便于区分连续四个实验 =====
  echo "---------- done Exchange_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 结束后打印完成提示，便于区分连续四个实验 =====
}

run_exchange 96
run_exchange 192
run_exchange 336
run_exchange 720
# ===== 修改结束：新增 Exchange 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

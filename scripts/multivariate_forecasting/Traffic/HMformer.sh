# ===== 修改开始：新增 Traffic 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，loss_mode 通过环境变量切换 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，loss_mode 通过环境变量切换 =====
# ===== 修改开始：新增 feedback 完整损失的 vol、bias、lag 权重和 soft lag 配置 =====
lambda_p=${LAMBDA_P:-1.0}
lambda_d=${LAMBDA_D:-0.1}
lambda_t=${LAMBDA_T:-0.5}
lambda_v=${LAMBDA_V:-0.1}
lambda_b=${LAMBDA_B:-0.1}
lambda_lag=${LAMBDA_LAG:-0.05}
lag_k=${LAG_K:-3}
lag_tau=${LAG_TAU:-0.1}
# ===== 修改开始：新增 feedback 静态/动态加权模式配置 =====
weight_mode=${WEIGHT_MODE:-static}
ema_beta=${EMA_BETA:-0.9}
weight_tau=${WEIGHT_TAU:-1.0}
if [ "${loss_mode}" = "baseline" ]; then
  exp_suffix=${loss_mode}
else
  exp_suffix=${loss_mode}_${weight_mode}
fi
# ===== 修改结束：新增 feedback 静态/动态加权模式配置 =====
# ===== 修改结束：新增 feedback 完整损失的 vol、bias、lag 权重和 soft lag 配置 =====
# ===== 修改开始：Traffic 维度较高，默认沿用参考脚本 batch_size=16 和 learning_rate=0.001 =====
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-16}
num_workers=${NUM_WORKERS:-8}
learning_rate=${LEARNING_RATE:-0.001}
# ===== 修改结束：Traffic 维度较高，默认沿用参考脚本 batch_size=16 和 learning_rate=0.001 =====
log_interval=${LOG_INTERVAL:-50}
percent=${PERCENT:-100}

# ===== 修改开始：打印脚本级实验配置，明确当前 Traffic 实验选择 =====
echo "========== HMformer Traffic 脚本配置 =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOSS_MODE=${loss_mode}"
# ===== 修改开始：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "LAMBDA_P=${lambda_p}, LAMBDA_D=${lambda_d}, LAMBDA_T=${lambda_t}"
echo "LAMBDA_V=${lambda_v}, LAMBDA_B=${lambda_b}, LAMBDA_LAG=${lambda_lag}"
echo "LAG_K=${lag_k}, LAG_TAU=${lag_tau}"
# ===== 修改开始：打印 feedback 静态/动态加权模式配置 =====
echo "WEIGHT_MODE=${weight_mode}, EMA_BETA=${ema_beta}, WEIGHT_TAU=${weight_tau}"
echo "EXP_SUFFIX=${exp_suffix}"
# ===== 修改结束：打印 feedback 静态/动态加权模式配置 =====
# ===== 修改结束：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "TRAIN_EPOCHS=${train_epochs}, BATCH_SIZE=${batch_size}, NUM_WORKERS=${num_workers}"
echo "LEARNING_RATE=${learning_rate}, PERCENT=${percent}, LOG_INTERVAL=${log_interval}"
echo "pred_len list: 96 192 336 720"
echo "=============================================="
# ===== 修改结束：打印脚本级实验配置，明确当前 Traffic 实验选择 =====

run_traffic() {
  pred_len=$1

  # ===== 修改开始：每个 horizon 开始前打印当前 Traffic 运行任务，便于对应日志和结果 =====
  echo "---------- start traffic_96_${pred_len}_${exp_suffix} ----------"
  # ===== 修改结束：每个 horizon 开始前打印当前 Traffic 运行任务，便于对应日志和结果 =====

  # ===== 修改开始：调用 main.py 时传入 feedback 完整损失参数 =====
  python -u main.py \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_${pred_len}_${exp_suffix} \
    --model HMformer \
    --data custom \
    --features M \
    --freq 0 \
    --percent ${percent} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 4 \
    --enc_in 862 \
    --c_out 862 \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --train_epochs ${train_epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --learning_rate ${learning_rate} \
    --log_interval ${log_interval} \
    --loss_mode ${loss_mode} \
    --lambda_p ${lambda_p} \
    --lambda_d ${lambda_d} \
    --lambda_t ${lambda_t} \
    --lambda_v ${lambda_v} \
    --lambda_b ${lambda_b} \
    --lambda_lag ${lambda_lag} \
    --lag_k ${lag_k} \
    --lag_tau ${lag_tau} \
    --weight_mode ${weight_mode} \
    --ema_beta ${ema_beta} \
    --weight_tau ${weight_tau}

  # ===== 修改结束：调用 main.py 时传入 feedback 完整损失参数 =====

  # ===== 修改开始：每个 horizon 结束后打印完成提示，便于区分连续四个 Traffic 实验 =====
  echo "---------- done traffic_96_${pred_len}_${exp_suffix} ----------"
  # ===== 修改结束：每个 horizon 结束后打印完成提示，便于区分连续四个 Traffic 实验 =====
}

run_traffic 96
run_traffic 192
run_traffic 336
run_traffic 720
# ===== 修改结束：新增 Traffic 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

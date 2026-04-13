# ===== 修改开始：新增 ETTh2 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，loss_mode 通过环境变量切换 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，loss_mode 通过环境变量切换 =====
lambda_p=${LAMBDA_P:-1.0}
lambda_d=${LAMBDA_D:-1.0}
lambda_t=${LAMBDA_T:-1.0}
# ===== 修改开始：完整测试默认使用更充分的 epoch、更大的 batch 和更多数据加载进程 =====
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-256}
num_workers=${NUM_WORKERS:-8}
# ===== 修改结束：完整测试默认使用更充分的 epoch、更大的 batch 和更多数据加载进程 =====
log_interval=${LOG_INTERVAL:-50}
percent=${PERCENT:-100}

# ===== 修改开始：打印脚本级实验配置，明确当前 ETTh2 实验选择 =====
echo "========== HMformer ETTh2 脚本配置 =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOSS_MODE=${loss_mode}"
echo "LAMBDA_P=${lambda_p}, LAMBDA_D=${lambda_d}, LAMBDA_T=${lambda_t}"
echo "TRAIN_EPOCHS=${train_epochs}, BATCH_SIZE=${batch_size}, NUM_WORKERS=${num_workers}"
echo "PERCENT=${percent}, LOG_INTERVAL=${log_interval}"
echo "pred_len list: 96 192 336 720"
echo "============================================"
# ===== 修改结束：打印脚本级实验配置，明确当前 ETTh2 实验选择 =====

run_etth2() {
  pred_len=$1
  d_model=$2
  d_ff=$3

  # ===== 修改开始：每个 horizon 开始前打印当前 ETTh2 运行任务，便于对应日志和结果 =====
  echo "---------- start ETTh2_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 开始前打印当前 ETTh2 运行任务，便于对应日志和结果 =====

  python -u main.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_${pred_len}_${loss_mode} \
    --model HMformer \
    --data ett_h \
    --features M \
    --freq 0 \
    --percent ${percent} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --enc_in 7 \
    --c_out 7 \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --itr 1 \
    --train_epochs ${train_epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --log_interval ${log_interval} \
    --loss_mode ${loss_mode} \
    --lambda_p ${lambda_p} \
    --lambda_d ${lambda_d} \
    --lambda_t ${lambda_t}

  # ===== 修改开始：每个 horizon 结束后打印完成提示，便于区分连续四个 ETTh2 实验 =====
  echo "---------- done ETTh2_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 结束后打印完成提示，便于区分连续四个 ETTh2 实验 =====
}

run_etth2 96 128 128
run_etth2 192 128 128
run_etth2 336 128 128
run_etth2 720 128 128
# ===== 修改结束：新增 ETTh2 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

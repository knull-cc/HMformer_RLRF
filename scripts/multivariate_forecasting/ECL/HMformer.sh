# ===== 修改开始：新增 ECL 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，loss_mode 通过环境变量切换 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，loss_mode 通过环境变量切换 =====
lambda_p=${LAMBDA_P:-1.0}
lambda_d=${LAMBDA_D:-1.0}
lambda_t=${LAMBDA_T:-1.0}
# ===== 修改开始：ECL 维度较高，默认沿用参考脚本 batch_size=16 和 learning_rate=0.0005 =====
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-16}
num_workers=${NUM_WORKERS:-8}
learning_rate=${LEARNING_RATE:-0.0005}
# ===== 修改结束：ECL 维度较高，默认沿用参考脚本 batch_size=16 和 learning_rate=0.0005 =====
log_interval=${LOG_INTERVAL:-50}
percent=${PERCENT:-100}

# ===== 修改开始：打印脚本级实验配置，明确当前 ECL 实验选择 =====
echo "========== HMformer ECL 脚本配置 =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOSS_MODE=${loss_mode}"
echo "LAMBDA_P=${lambda_p}, LAMBDA_D=${lambda_d}, LAMBDA_T=${lambda_t}"
echo "TRAIN_EPOCHS=${train_epochs}, BATCH_SIZE=${batch_size}, NUM_WORKERS=${num_workers}"
echo "LEARNING_RATE=${learning_rate}, PERCENT=${percent}, LOG_INTERVAL=${log_interval}"
echo "pred_len list: 96 192 336 720"
echo "=========================================="
# ===== 修改结束：打印脚本级实验配置，明确当前 ECL 实验选择 =====

run_ecl() {
  pred_len=$1

  # ===== 修改开始：每个 horizon 开始前打印当前 ECL 运行任务，便于对应日志和结果 =====
  echo "---------- start ECL_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 开始前打印当前 ECL 运行任务，便于对应日志和结果 =====

  python -u main.py \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_${pred_len}_${loss_mode} \
    --model HMformer \
    --data custom \
    --features M \
    --freq 0 \
    --percent ${percent} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --enc_in 321 \
    --c_out 321 \
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
    --lambda_t ${lambda_t}

  # ===== 修改开始：每个 horizon 结束后打印完成提示，便于区分连续四个 ECL 实验 =====
  echo "---------- done ECL_96_${pred_len}_${loss_mode} ----------"
  # ===== 修改结束：每个 horizon 结束后打印完成提示，便于区分连续四个 ECL 实验 =====
}

run_ecl 96
run_ecl 192
run_ecl 336
run_ecl 720
# ===== 修改结束：新增 ECL 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

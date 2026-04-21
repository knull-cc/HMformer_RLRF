# ===== 修改开始：新增 Weather 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
set -e
# ===== 修改开始：开启 pipefail，确保 tee 同步日志时 main.py 失败会让脚本退出 =====
set -o pipefail
# ===== 修改结束：开启 pipefail，确保 tee 同步日志时 main.py 失败会让脚本退出 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，loss_mode 通过环境变量切换 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，loss_mode 通过环境变量切换 =====
# ===== 修改开始：新增自蒸馏环境变量，支持直接关闭或加载离线 teacher =====
distill_mode=${DISTILL_MODE:-none}
teacher_path=${TEACHER_PATH:-}
teacher_path_template=${TEACHER_PATH_TEMPLATE:-}
distill_alpha=${DISTILL_ALPHA:-0.1}
distill_start_epoch=${DISTILL_START_EPOCH:-1}
# ===== 修改结束：新增自蒸馏环境变量，支持直接关闭或加载离线 teacher =====
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
# ===== 修改开始：为蒸馏实验追加独立后缀，避免和非蒸馏 checkpoint 混淆 =====
if [ "${distill_mode}" != "none" ]; then
  distill_alpha_tag=$(printf '%s' "${distill_alpha}" | tr '.' 'p')
  exp_suffix=${exp_suffix}_${distill_mode}_da${distill_alpha_tag}
fi
# ===== 修改结束：为蒸馏实验追加独立后缀，避免和非蒸馏 checkpoint 混淆 =====
# ===== 修改结束：新增 feedback 静态/动态加权模式配置 =====
# ===== 修改结束：新增 feedback 完整损失的 vol、bias、lag 权重和 soft lag 配置 =====
# ===== 修改开始：Weather 使用 21 变量，默认给出完整测试训练轮数和可覆盖的训练参数 =====
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-256}
num_workers=${NUM_WORKERS:-8}
learning_rate=${LEARNING_RATE:-0.0001}
# ===== 修改结束：Weather 使用 21 变量，默认给出完整测试训练轮数和可覆盖的训练参数 =====
log_interval=${LOG_INTERVAL:-50}
percent=${PERCENT:-100}

# ===== 修改开始：打印脚本级实验配置，明确当前 Weather 实验选择 =====
echo "========== HMformer Weather 脚本配置 =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOSS_MODE=${loss_mode}"
# ===== 修改开始：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "LAMBDA_P=${lambda_p}, LAMBDA_D=${lambda_d}, LAMBDA_T=${lambda_t}"
echo "LAMBDA_V=${lambda_v}, LAMBDA_B=${lambda_b}, LAMBDA_LAG=${lambda_lag}"
echo "LAG_K=${lag_k}, LAG_TAU=${lag_tau}"
# ===== 修改开始：打印 feedback 静态/动态加权模式配置 =====
echo "WEIGHT_MODE=${weight_mode}, EMA_BETA=${ema_beta}, WEIGHT_TAU=${weight_tau}"
echo "EXP_SUFFIX=${exp_suffix}"
# ===== 修改开始：打印蒸馏配置，便于平滑区分蒸馏与非蒸馏实验 =====
echo "DISTILL_MODE=${distill_mode}, DISTILL_ALPHA=${distill_alpha}, DISTILL_START_EPOCH=${distill_start_epoch}"
echo "TEACHER_PATH=${teacher_path}"
echo "TEACHER_PATH_TEMPLATE=${teacher_path_template}"
# ===== 修改结束：打印蒸馏配置，便于平滑区分蒸馏与非蒸馏实验 =====
# ===== 修改结束：打印 feedback 静态/动态加权模式配置 =====
# ===== 修改结束：打印 feedback 完整损失的六类权重和 soft lag 配置 =====
echo "TRAIN_EPOCHS=${train_epochs}, BATCH_SIZE=${batch_size}, NUM_WORKERS=${num_workers}"
echo "LEARNING_RATE=${learning_rate}, PERCENT=${percent}, LOG_INTERVAL=${log_interval}"
echo "pred_len list: 96 192 336 720"
echo "=============================================="
# ===== 修改结束：打印脚本级实验配置，明确当前 Weather 实验选择 =====

# ===== 修改开始：蒸馏模式下校验 teacher 路径输入，避免 main.py 运行到中途才报错 =====
if [ "${distill_mode}" != "none" ] && [ -z "${teacher_path}" ] && [ -z "${teacher_path_template}" ]; then
  echo "ERROR: DISTILL_MODE=${distill_mode} 但 TEACHER_PATH 和 TEACHER_PATH_TEMPLATE 都为空。"
  exit 1
fi
# ===== 修改结束：蒸馏模式下校验 teacher 路径输入，避免 main.py 运行到中途才报错 =====

# ===== 修改开始：新增 horizon 结果缓存和等权平均打印函数，只解析日志不写 CSV =====
result_pred_lens=()
result_mses=()
result_maes=()

record_result() {
  pred_len=$1
  log_file=$2
  result_line=$(grep -E 'mae:[0-9.eE+-]+, mse:[0-9.eE+-]+' "${log_file}" | tail -n 1 || true)
  if [ -z "${result_line}" ]; then
    echo "WARNING: pred_len=${pred_len} 未解析到 mae/mse，跳过平均汇总。"
    return 0
  fi

  mae=$(echo "${result_line}" | sed -E 's/.*mae:([0-9.eE+-]+), mse:([0-9.eE+-]+).*/\1/')
  mse=$(echo "${result_line}" | sed -E 's/.*mae:([0-9.eE+-]+), mse:([0-9.eE+-]+).*/\2/')
  result_pred_lens+=("${pred_len}")
  result_mses+=("${mse}")
  result_maes+=("${mae}")
}

print_average_results() {
  if [ ${#result_mses[@]} -eq 0 ]; then
    echo "========== HMformer 四个预测长度等权平均结果 =========="
    echo "未解析到任何 mse/mae 结果，无法计算平均值。"
    echo "======================================================"
    return 0
  fi

  echo "========== HMformer 四个预测长度等权平均结果 =========="
  for idx in "${!result_pred_lens[@]}"; do
    printf 'pred_len: %-4s | mse: %.4f | mae: %.4f\n' "${result_pred_lens[$idx]}" "${result_mses[$idx]}" "${result_maes[$idx]}"
  done
  echo "------------------------------------------------------"
  avg_mse=$(printf '%s\n' "${result_mses[@]}" | awk '{sum += $1} END {if (NR > 0) printf "%.4f", sum / NR}')
  avg_mae=$(printf '%s\n' "${result_maes[@]}" | awk '{sum += $1} END {if (NR > 0) printf "%.4f", sum / NR}')
  echo "Average MSE: ${avg_mse}"
  echo "Average MAE: ${avg_mae}"
  echo "======================================================"
}
# ===== 修改结束：新增 horizon 结果缓存和等权平均打印函数，只解析日志不写 CSV =====

run_weather() {
  pred_len=$1

  # ===== 修改开始：每个 horizon 开始前打印当前 Weather 运行任务，便于对应日志和结果 =====
  echo "---------- start weather_96_${pred_len}_${exp_suffix} ----------"
  # ===== 修改结束：每个 horizon 开始前打印当前 Weather 运行任务，便于对应日志和结果 =====

  # ===== 修改开始：按 pred_len 解析当前 horizon 的 teacher checkpoint 路径，支持单一路径或模板路径 =====
  run_teacher_path=${teacher_path}
  if [ -n "${teacher_path_template}" ]; then
    run_teacher_path=${teacher_path_template//\{pred_len\}/${pred_len}}
  fi
  # ===== 修改结束：按 pred_len 解析当前 horizon 的 teacher checkpoint 路径，支持单一路径或模板路径 =====

  # ===== 修改开始：调用 main.py 时传入 feedback 完整损失参数，并同步捕获日志用于平均值汇总 =====
  run_log=$(mktemp)
  python -u main.py \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_${pred_len}_${exp_suffix} \
    --model HMformer \
    --data custom \
    --features M \
    --freq 0 \
    --percent ${percent} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --enc_in 21 \
    --c_out 21 \
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
    --weight_tau ${weight_tau} \
    --distill_mode ${distill_mode} \
    --teacher_path "${run_teacher_path}" \
    --distill_alpha ${distill_alpha} \
    --distill_start_epoch ${distill_start_epoch} 2>&1 | tee "${run_log}"
  record_result "${pred_len}" "${run_log}"
  rm -f "${run_log}"

  # ===== 修改结束：调用 main.py 时传入 feedback 完整损失参数，并同步捕获日志用于平均值汇总 =====

  # ===== 修改开始：每个 horizon 结束后打印完成提示，便于区分连续四个 Weather 实验 =====
  echo "---------- done weather_96_${pred_len}_${exp_suffix} ----------"
  # ===== 修改结束：每个 horizon 结束后打印完成提示，便于区分连续四个 Weather 实验 =====
}

run_weather 96
run_weather 192
run_weather 336
run_weather 720
# ===== 修改开始：四个 horizon 全部完成后打印等权平均 MSE/MAE =====
print_average_results
# ===== 修改结束：四个 horizon 全部完成后打印等权平均 MSE/MAE =====
# ===== 修改结束：新增 Weather 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

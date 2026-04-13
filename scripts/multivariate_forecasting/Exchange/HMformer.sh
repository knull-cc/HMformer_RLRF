# ===== 修改开始：新增 Exchange 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== 修改开始：只运行 HMformer，删除可切换模型名变量 =====
loss_mode=${LOSS_MODE:-baseline}
# ===== 修改结束：只运行 HMformer，删除可切换模型名变量 =====
lambda_p=${LAMBDA_P:-1.0}
lambda_d=${LAMBDA_D:-1.0}
lambda_t=${LAMBDA_T:-1.0}
train_epochs=${TRAIN_EPOCHS:-1}
# ===== 修改开始：Exchange 720 horizon 需要完整训练集窗口，避免 percent=10 导致 __len__ 为负 =====
percent=${PERCENT:-100}
# ===== 修改结束：Exchange 720 horizon 需要完整训练集窗口，避免 percent=10 导致 __len__ 为负 =====

run_exchange() {
  pred_len=$1

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
    --batch_size 32 \
    --num_workers 2 \
    --loss_mode ${loss_mode} \
    --lambda_p ${lambda_p} \
    --lambda_d ${lambda_d} \
    --lambda_t ${lambda_t}
}

run_exchange 96
run_exchange 192
run_exchange 336
run_exchange 720
# ===== 修改结束：新增 Exchange 多变量预测 HMformer 测试脚本，支持通过环境变量切换 loss 实验 =====

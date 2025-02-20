source venv/bin/activate
ts=$(date +%Y%m%d_%H%M%S)

accelerate launch --num_processes 1 --config_file configs/one_gpu.yaml grpo_gsm8k.py \
    --output_dir outputs/Qwen2.5-1.5B-Instruct.GRPO \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 20 \
    --learning_rate 3e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --num_generations 3 \
    --save_steps 50 \
    --max_steps 1000 \
    --torch_dtype bfloat16 \
    --bf16 \
    > "train_logs_${ts}.out" 2>&1 < /dev/null
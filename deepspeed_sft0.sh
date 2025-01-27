WANDB_TOKEN="7980d52e9595b40cca88d26f98d771a9d714b20f"
HF_TOKEN="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 "

model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B"
embedding_model_id="BAAI/bge-m3"
instruction_template="/home/user09/beaver/instruction_templates/sft_v0.9.txt"
deepspeed_config="deepspeed/config_zero3.json"
learning_rate_list="3e-6"
gradient_accumulation_steps_list="8"

_2=2
for learning_rate in $learning_rate_list
do
    for gradient_accumulation_steps in $gradient_accumulation_steps_list
    do
    deepspeed --include=localhost:0 --master_port 60000 instruction_tuning/sft.py \
        --WANDB_TOKEN $WANDB_TOKEN \
        --HF_TOKEN $HF_TOKEN \
        --project beaver_sft_full \
        --model_name_or_path $model_id \
        --embedding_model $embedding_model_id \
        --instruction_template $instruction_template \
        --do_train \
        --do_eval \
        --do_predict \
        --train_file /home/user06/beaver/data/dataset/beaver_v12.3_train.json \
        --validation_file /home/user06/beaver/data/dataset/beaver_v12.3_val.json \
        --output_dir /home/user06/beaver/log/sft \
        --overwrite_output_dir \
        --save_total_limit 2 \
        --learning_rate $learning_rate \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --eval_steps 10 \
        --save_steps 100 \
        --logging_steps 25 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --max_steps 500 \
        --report_to wandb \
        --is_lora False \
        --retriever_k 4 \
        --retriever_bert_weight 0.7 \
        --seed 42 \
        --bf16 \
        --deepspeed $deepspeed_config
    done
done
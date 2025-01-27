WANDB_TOKEN="7980d52e9595b40cca88d26f98d771a9d714b20f"
HF_TOKEN="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 "

model_id="beomi/Llama-3-Open-Ko-8B-Instruct-preview"
embedding_model_id="BAAI/bge-m3"
instruction_template="instruction_templates/sft_v0.5.txt"
learning_rate_list="2e-5"
gradient_accumulation_steps_list="16"
retriever_list="True"

_2=2
for learning_rate in $learning_rate_list
do
    for gradient_accumulation_steps in $gradient_accumulation_steps_list
    do
        for retriever in $retriever_list
        do
        CUDA_VISIBLE_DEVICES=1 python instruction_tuning/sft.py \
            --WANDB_TOKEN $WANDB_TOKEN \
            --HF_TOKEN $HF_TOKEN \
            --cache_dir /nas/.cache/huggingface \
            --project beaver_sft_lora \
            --model_name_or_path $model_id \
            --embedding_model $embedding_model_id \
            --instruction_template $instruction_template \
            --do_train \
            --do_eval \
            --do_predict \
            --train_file /home/user06/beaver/data/dataset/beaver_v8.3_train.json \
            --validation_file /home/user06/beaver/data/dataset/beaver_v8.3_val.json \
            --output_dir /home/user06/beaver/log/capstone_sft \
            --overwrite_output_dir \
            --save_total_limit 2 \
            --learning_rate $learning_rate \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --lr_scheduler_type linear \
            --warmup_ratio 0.1 \
            --weight_decay 0.01 \
            --eval_steps 50 \
            --save_steps 50 \
            --logging_steps 5 \
            --evaluation_strategy steps \
            --load_best_model_at_end \
            --max_steps 500 \
            --report_to wandb \
            --is_lora True \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.0 \
            --lora_target_modules "q_proj,v_proj" \
            --is_retriever $retriever \
            --retriever_k 4 \
            --retriever_bert_weight 0.7 \
            --seed 42 \
            --bf16 
        done
    done
done
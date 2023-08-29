CUDA_VISIBLE_DEVICES=0,2 python3 GPT2.py \
    --pretrained_model ../model/pretrained_GPT2 \
    --epoch 20 \
    --batch_size 1 \
    --lr 5e-6 \
    --weight_decay 0.001\
    --lr_scheduler_type cosine \
    --warmup_ratio 0.3 \
    --output_dir ../model \
    --gradient_accumulation_steps 4 \
    
CUDA_VISIBLE_DEVICES=1 python3 BERT_NSP.py \
    --pretrained_model ../model/pretrained_NSP \
    --epoch 25 \
    --batch_size 8 \
    --lr 5e-6 \
    --weight_decay 0.001\
    --lr_scheduler_type cosine \
    --warmup_ratio 0.3 \
    --train_addr ./data/train_NSP.json \
    --valid_addr ./data/valid_NSP.json \
    --test_addr ./data/test_NSP.json \
    --output_dir ../model2 \
    
    

    

    
    
    
    
#python3 GPT2.py \
#    --pretrained_model ../model/checkpoint-98082 \
#    --epoch 20 \
#    --batch_size 8 \
#    --lr 5e-6 \
#    --weight_decay 0.001\
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0.3 \
#    --test
    
    
    
    

#scheduler: linear, cosine, cosine with restarts, polynimial, constant, constant_with_warmup

#best: train acc = 0.9348 --> 87570, test acc = 0.9346


#base case test acc: 0.6348, large acc: 0.652, roberta 0.5

#old version test acc= 0.9077

#model candidate: xlnet, roberta
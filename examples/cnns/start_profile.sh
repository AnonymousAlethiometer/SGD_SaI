#!/bin/bash
###
 # @Author: Unknown
 # @Date: 2024-11-11 13:02:26
 # @LastEditTime: 2024-11-11 22:09:33
 # @LastEditors: Unknown
 # @Description: 
 # @FilePath: /Unknown/start_profile.sh
### 


bs=1
cuda_device=0
num_iters=20
model_name=vit_s_16
root_path=./profiles__p2_${model_name}
# context_length=1024
context_length=-1

# CUDA_VISIBLE_DEVICES=$cuda_device python profile_speed.py --num_iters $num_iters --optimizer $optimizer --batch_size $bs --device cuda:0 --model_name $model_name --root_path $root_path --context_length $context_length
opts=(SGD_sai Adam SGD AdamW Prodigy)
# opts=(Prodigy)
for opt in ${opts[@]}
do
    CUDA_VISIBLE_DEVICES=$cuda_device python profile_speed.py --num_iters $num_iters --optimizer $opt --batch_size $bs --device cuda:0 --model_name $model_name --root_path $root_path --context_length $context_length
done



model_name=vit_h_14
root_path=./profiles__p2_${model_name}
context_length=-1
opts=(SGD_sai Adam SGD AdamW Prodigy)
for opt in ${opts[@]}
do
    CUDA_VISIBLE_DEVICES=$cuda_device python profile_speed.py --num_iters $num_iters --optimizer $opt --batch_size $bs --device cuda:0 --model_name $model_name --root_path $root_path --context_length $context_length
done




model_name=gpt2
root_path=./profiles__p2_${model_name}
context_length=1024
opts=(SGD_sai Adam SGD AdamW Prodigy)
for opt in ${opts[@]}
do
    CUDA_VISIBLE_DEVICES=$cuda_device python profile_speed.py --num_iters $num_iters --optimizer $opt --batch_size $bs --device cuda:0 --model_name $model_name --root_path $root_path --context_length $context_length
done



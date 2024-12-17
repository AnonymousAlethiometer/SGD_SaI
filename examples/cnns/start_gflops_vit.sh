#!/bin/bash
###
 # @Author: Unknown
 # @Date: 2024-11-14 04:53:34
 # @LastEditTime: 2024-11-14 06:22:39
 # @LastEditors: Unknown
 # @Description: 
 # @FilePath: /Unknown/start_gflops_vit.sh
### 



bs=1
context_length=1024
device=cuda:0
num_iters=3

model_names=('vit_s_16' 'vit_h_14')
save_path=./profiles__${model_name}




opts=('Ours' 'Adam' 'AdamW' 'SGD' 'Prodigy' 'Adam_mini')
iters=(1 2 3 4 5 6 7 8 9 10)
for model_name in ${model_names[@]}
do
    save_path=./profiles__${model_name}_gflops.csv

    for opt in ${opts[@]}
    do
        echo "Optimizing: $opt"
        for iter in ${iters[@]}
        do
            echo "Iterating: $iter"
            python gflops_vit.py --num_iters 1 --batch_size $bs --device $device --model $model_name --save_to $save_path --opt $opt
        done
    done
done

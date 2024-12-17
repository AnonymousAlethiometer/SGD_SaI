#!/bin/bash
###
 # @Author: Unknown
 # @Date: 2024-11-14 04:53:34
 # @LastEditTime: 2024-11-14 05:08:50
 # @LastEditors: Unknown
 # @Description: 
 # @FilePath: /Unknown/start_gflops.sh
### 



bs=1
context_length=1024
device=cuda:0
num_iters=3

model_name=gpt2_xl
save_path=./profiles__${model_name}_gflops.csv


opts=('Ours' 'Adam' 'AdamW' 'SGD' 'Prodigy' 'Adam_mini')
iters=(1 2 3 4 5 6 7 8 9 10)
for opt in ${opts[@]}
do
    echo "Optimizing: $opt"
    for iter in ${iters[@]}
    do
        echo "Iterating: $iter"
        python gflops_gpt2.py --num_iters 1 --batch_size $bs --device $device --model $model_name --save_to $save_path --context_length $context_length --opt $opt
    done
done

#!/bin/bash
###
 # @Author: Unknown
 # @Date: 2024-05-14 23:18:30
 # @LastEditTime: 2024-08-10 17:50:27
 # @LastEditors: Unknown
 # @Description: grid search for base models of ResNet18 on CIFAR-10, then it would be used as the baseline for further comparison
 # @FilePath: /Unknown/scripts/grid_search_base.sh
### 

gpu=0
seed=42
optimizers=(Adam-gs AdamW-gs AdamS-gs SGD_sai)

wd_list=(5e-5 5e-4 5e-3 5e-2 5e-1)
lr_initials=(1e-4 1e-3 1e-2 1e-1)
root_save_dir="logs_final_gs"


batch_size=128
num_workers=8
epochs=200

# if [ $optimizer == "SGD" ] || [ $optimizer == "SGDS" ]
# then
#     lr_initial=1e-1
# else
#     lr_initial=1e-3
# fi
# if [ $optimizer == "AdamSNR_GNSNRAN" ]
# then
#     wd_list=(1e-3)
# else
#     wd_list=(1e-3 5e-4 1e-4)
# fi


for optimizer in ${optimizers[@]}
do
    for lr_initial in ${lr_initials[@]}
    do
        for weight_decay in ${wd_list[@]}
        do
            dataset=cifar10
            lr_scheduler=SWDLambdaLR
            model=resnet18
            python vanilla_baseline_train.py --lr $lr_initial --weight_decay $weight_decay --save_path ./$root_save_dir/$dataset/$model-O_$optimizer-LR_$lr_initial-LS_$lr_scheduler-D_$dataset-S_$seed --optimizer $optimizer --model $model --seed $seed --gpu $gpu --dataset $dataset --data_aug --batch_size $batch_size --epochs $epochs --lr_scheduler $lr_scheduler --num_workers=$num_workers
        done
    done
done

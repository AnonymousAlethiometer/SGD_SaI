# SGD-SaI

This repository contains the official PyTorch implementation of the paper <em>"No More Adam: Learning Rate Scaling at Initialization is All You Need"</em> at [arxiv](https://arxiv.org/abs/2412.11768), providing a runtime and memory efficient optimizer for training large language models and diffusion models, named **"SGD-SaI"**.

In this work, we question the necessity of adaptive gradient methods for training deep neural networks. To address this, we introduce SGD-SaI, a simple yet effective enhancement to stochastic gradient descent with momentum (SGDM). SGD-SaI assigns distinct preconditioned learning rate scales to various parameter groups based on their initial gradient signal-to-noise ratios (g-SNR). Unlike Adam-like methods that depend on second-order momentum to adjust learning rates in response to imbalances detected in the gradient norm history, g-SNR enables us to identify gradient variance prior to the start of training. This proactive adjustment of learning rate scales minimizes the variance of parameter updates throughout the training process.

<!-- [Overview](./figures/overview.svg) -->
<p align='center'>
    <img src="./figures/algorithm_overview.svg" height='300px'/>
    <br/>
    <em><b>Figure1:</b>This graph illustrates the differences in local gain behaviours exhibited by four optimizers throughout the training process. We present two popular adaptive gradient methods: Adam(W) and the memory-efficient Adam-mini. The local gains for these methods are recalculated continuously at each step based on the gradients. In contrast, SGD and SGD-SaI are both non-adaptive methods, meaning their local gains remain fixed throughout the training.</em>
</p>

### Results
<p align='center'>
    <img src="./figures/combine_3d_scatter_v3.svg"/>
    <br/>
    <em>
    <b>Figure2:</b>This figure displays the results from a grid search conducted on the classic ResNet18 model using the CIFAR10 dataset and ViT-S/16 model using the ImageNet-1K dataset. The maximum top-1 test accuracy is highlighted in red text. Our method surpasses other popular optimizers, achieving the highest test accuracy in traditional CNNs. Additionally, our method demonstrates superior performance compared to other popular optimizers, particularly in terms of stability in response to changes in hyperparameters.
    </em>
</p>
<p align='center'>
    <img src="./figures/pretain_results.svg"/>
    <br/>
    <em>
    <b>Figure3:</b> This figures shows the pre-train results on GPT2 and ViT. Our method save 50% memory usage of the state tensors of AdamW, and it is 3x faster than AdamMini. And for ViT, although our method has a slower convergence speed, we can still achieve comparable performance by the end of the training process. Additionally, our approach is designed to have a lower memory footprint and a faster optimization speed.
    </em>
</p>

### Efficiency

<p align='center'>
    <img src="./figures/speed_memory_growth.svg"/>
    <br/>
    <em>
    <b>Figure4:</b>  The chart illustrates how memory usage and optimizer step time (in wall-clock time) increase with larger model sizes. It highlights the substantial memory overhead of storing optimizer states as model sizes grow. SGD-SaI exhibits significantly lower memory usage than AdamW and has the shortest optimization step runtime. This runtime refers to the wall clock time required for the optimizer step function. All statistics were measured on a single NVIDIA A100-80GB. 
    </em>
</p>


<p align='center'>
    <img src="./figures/algorithm_pseudocode.png" height='450px'/>
    <br/>
    <em>
    <b>Figure5:</b> The pseudocode of the SGD-SaI optimizer. 
    </em>
</p>

<!-- [Memory Comparison](./figures/optimizer_memory_comparison.svg) -->
<!-- <img src="./figures/optimizer_memory_comparison.svg"/> -->
<!-- [Stability & Perfomance with Hyperparameters Changes](./figures/3d_scatter.svg) -->
<!-- <img src="./figures/3d_scatter.svg" /> -->

<!-- <img src='./figures/algorithm_pseudocode.png'/> -->

## How To Use

### Installation
Prerequisites:
- Python >= 3.6
- PyTorch >= 1.7.0

Since most of this optimizer uses the native PyTorch APIs, it should have a wider compatibility with different versions of PyTorch. However, we recommend using the Pytorch 2.X version for better performance and compatibility.

Install from PyPI:
```bash
pip install sgd-sai
```

Install from the source code:
```bash
git clone https://github.com/AnonymousAlethiometer/SGD_SaI.git

cd SGD_SaI

# you can use the flag "--use-feature=in-tree-build" to avoid the warning of "FutureWarning: The 'build' command is deprecated"
pip install . --use-feature=in-tree-build

# [Optional] Or you can use '-e' flag to install in editable mode
pip install -e . --use-feature=in-tree-build
```


### Usage of the optimizer:

The optimizer is normally used in the following way:

```python
from sgd_sai import SGD_sai

# initialize the optimizer
optimizer = SGD_sai(model.parameters(), lr=lr, momentum=0.9, eps=1e-08, weight_decay=weight_decay)

for _ in range(steps):
    pred = model(input_ids)
    loss = loss_fn(pred, labels)
    # calculate the gradient
    loss.backward()
    # From version 1.0.3, we do not need to call the warmup_step() manually, it will be called automatically after the first gradient calculation. Rightnow, it is user-friendly for different dirstributed training frameworks.
    # # process the warmup step, only need once after the gradient is calculated
    # # if not hasattr(optimizer, 'has_warmup') and hasattr(optimizer, 'warmup_step'):
    # #     optimizer.warmup_step()
    # #     optimizer.has_warmup = True
    # # update the parameters
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### Distributed Training
**Version before: 1.0.3.**

For distributed training, you need to ensure to perform this g-SNR calculation (refer as the `.warmup step()`) on each worker. Even if you accidentally perform it multiple times, it will not affect the final result thanks to the stability of the g-SNR values. Feel free to use the optimizer in your own training scripts. 

In most circumstances, you only need to replace the original optimizer with our optimizer, perform the `.warmup step()` after first gradient calculation (aka. the first effective invoke of `loss.backwards()`) and keep the rest of the code unchanged.

**Version after: 1.0.3.**

Nothing need to be changed, the optimizer will automatically perform the warmup step after the first gradient calculation. Juts use it like a normal Pytorch optimizer.


## Example:

The CNN examples lie in the `examples` directory. It contains the training code for CNN models, as well as the profiling code for the optimizer perfomance evaluation.

Please follow the README in that directory will guide you to restore the environment. Due to the procedure of anonymization, although the main part has been leaved unchanged, some of the components may not be available, try to delete the unavailable resources as needed. 

The ViT example will be released soon.


## Acknowledgement
1. The codebase is based on the [timm:pytorch-image-models](https://github.com/huggingface/pytorch-image-models)(ViT training example, release soon), [NanoGPT](https://github.com/karpathy/nanoGPT) and [Adam-mini](https://github.com/zyushun/Adam-mini)(GPT2 training) repository.

2. We thanks for [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for an accurate and efficient way to profile the memory usage of the optimizer.


## Citation
If you find this work helpful, please consider citing our paper:
```
@article{xu2024adamlearningratescaling,
    title={No More Adam: Learning Rate Scaling at Initialization is All You Need}, 
    author={Minghao Xu and Lichuan Xiang and Xu Cai and Hongkai Wen},
    year={2024},
    eprint={2412.11768},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2412.11768}, 
}
```

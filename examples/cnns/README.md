<!--
 * @Author: Unknown
 * @Date: 2024-03-01 10:01:38
 * @LastEditTime: 2024-11-11 23:02:12
 * @LastEditors: Unknown
 * @Description: 
 * @FilePath: /Unknown/README.md
-->
# Unknown


## setup environment
install from scratch
```bash
    # conda create -n ml python=3.10.4
    conda create -n ml python=3.11.4
    conda activate ml
    
    # install pytorch
    # # conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c nvidia
    # conda install pytorch==2.0.1 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c nvidia
    # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge #official
    # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.7 -c pytorch -c conda-forge
    # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda==11.7 -c pytorch -c nvidia
    # or you may use a old cuda version, in this case, its 11.3
    # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 

    # install common sci-computing packages and dependencies
    # pip install pandas numpy matplotlib einops tensorboard lightning torchmetrics torchattacks

    pip install torchattacks==3.5.1 einops==0.7.0 pyyaml==6.0.1 tensorboard==2.16.2 torcheval==0.0.7 pandas==2.2.1 h5py==3.10.0 accelerate==0.29.1 wandb==0.16.6

    conda env export > environment.yaml
    # conda env create -n ml -f environment.yaml

    # # install common sci-computing packages and dependencies
    # pip install pandas numpy matplotlib
    
    # # install einops
    # pip install einops

    # # install tensorboard to activate lightning tensorboard recording
    # pip install tensorboard

    # # install pytorch lightning
    # pip install lightning
    # pip install torchmetrics
```
install from environment file
```bash
    conda env create -n ml -f environment.yaml
```


## Setup Sulis

```bash
# module load GCCcore/11.3.0 Python/3.10.4 
# module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.12.1-CUDA-11.7.0
module load PyTorch/2.0-Miniconda3-4.12.0
```


## Run

1. For single gpu training, modify the shell script `./scripts/baseline_single_gpu.sh` to set the `--gpu` and `--optimizer` path, then run the following command:
```bash
    cd $PROJECT_DIR$
    ./scripts/baseline_single_gpu.sh
```

2. For multi gpu training, modify the shell script `./scripts/accelerate_config.yaml` of use "accelerate" command to auto generate the default config and replace the content of `./scripts/accelerate_config.yaml`, then run the following command:
```bash
    cd $PROJECT_DIR$
    accelerate config
```
launch using 🤗 accelerate command
```bash
    cd $PROJECT_DIR$
    ./scripts/accelerate_launch.sh
```

## Misc for Slurm
### 1. Submit Job to Slurm
```bash
    cd $PROJECT_DIR$
    # submit job to slurm
    sbatch ./scripts/slurm_job.sh
```

### 1.1 Generate batch files(sbatch) for Slurm
```bash
    cd $PROJECT_DIR$
    # generate batch files for slurm
    python ./scripts/generate_slurm_scripts.py
```

### 2. Check Job Status
```bash
    # check job status
    squeue --me
```

### 3. Cancel Job
```bash
    # cancel job, using squeue to get the job id
    scancel <job_id>
```

## Misc for Others
### 1. Accumulate and generate the report
```bash
    cd $PROJECT_DIR$
    # accumulate the report
    python ./scripts/generate_exp_reports.py
```

### 1.1 If robust acc is not calculateded, run the following command to calculate the robust acc
```bash
    cd $PROJECT_DIR$
    # accumulate the report
    python ./scripts/complete_adv_progress_if_not_exists.py
```

tar -cvf - ./../logs_final/ | split --bytes=2G --suffix-length=4 --numeric-suffixes - logs_final.tar.
cat logs_final.tar.* | tar -xvf -

```bash
    cd $PROJECT_DIR$
    # accumulate the report
    python ./scripts/generate_exp_reports.py
```


wandb login
python -m wandb login

conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name


# vis env only for plotting

conda create -n vis python=3.11.4
conda activate vis
pip install pandas numpy matplotlib plotly pyyaml 


# Profiling the optimizers. We need to use a up-to-date version of pytorch, which is 2.4 for all api to be available
We need to reinstall the whole environment with the following command
```bash
    conda create -n torch_profile python=3.11.4
    conda activate torch_profile
    # conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install matplotlib einops tensorboard pandas
```
Then we will unlock the complete profiling ability of the pytorch, and we can use the following command to profile the optimizer
```bash
    python profile_2.py
    ./start_profile.sh
```


Remember to change the default profiler API: 
/home/Unknown/miniconda3/envs/torch_profile/lib/python3.11/site-packages/torch/profiler/_memory_profiler.py #method: export_memory_timeline_html

```python

'''/home/Unknown/miniconda3/envs/torch_profile/lib/python3.11/site-packages/torch/profiler/profiler.py #_KinetoProfile'''

class _KinetoProfile:
    ...
    def export_memory_timeline(self, path: str, device: Optional[str] = None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None and self.use_device and self.use_device != "cuda":
            device = self.use_device + ":0"

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".pdf"):
            self.mem_tl.export_memory_timeline_pdf(path, device)
        elif path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
            fp.close()
            if path.endswith("raw.json.gz"):
                self.mem_tl.export_memory_timeline_raw(fp.name, device)
            else:
                self.mem_tl.export_memory_timeline(fp.name, device)
            with open(fp.name) as fin:
                with gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
        else:
            self.mem_tl.export_memory_timeline(path, device)
    ...


''''/home/Unknown/miniconda3/envs/torch_profile/lib/python3.11/site-packages/torch/profiler/_memory_profiler.py #MemoryProfileTimeline'''
    def export_memory_timeline_pdf(
        self, path, device_str, figsize=(20, 12), title=None
    ) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        # Check if user has matplotlib installed, return gracefully if not.
        import importlib.util

        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is None:
            print(
                "export_memory_timeline_html failed because matplotlib was not found."
            )
            return

        from base64 import b64encode
        from os import remove
        from tempfile import NamedTemporaryFile

        import matplotlib.pyplot as plt
        import numpy as np

        mt = self._coalesce_timeline(device_str)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        # For this timeline, start at 0 to match Chrome traces.
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024**3
        device = torch.device(device_str)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)


        rc = {
            # 'figure.figsize':(10,5),
            'axes.facecolor':'white',
            'axes.grid' : True,
            'grid.color': '.9',
            #   'font.family':'Times New Roman',
            'font.size' : 20}
        plt.rcParams.update(rc)
        # sns.set(font_scale=2)  # crazy big font


        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=figsize, dpi=300)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )
        fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
        # Usually training steps are in magnitude of ms.
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join(
            ([title] if title else [])
            + [
                f"Max memory allocated: {max_memory_allocated/(1024**3):.2f} GiB \n"
                f"Max memory reserved: {max_memory_reserved/(1024**3):.2f} GiB"
            ]
        )
        axes.set_title(title)

        fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
```
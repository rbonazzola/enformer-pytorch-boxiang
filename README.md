PyTorch implementation of DeepMind's Enformer model.

The model implementation is by [Phil Wang](https://github.com/lucidrains) and [Boxiang Liu](https://github.com/boxiangliu/). I added the data pre-processing and training scripts



# Setup

This package has the following dependencies:

```
python==3.11.8
einops
torch==2.3.0
numpy
tqdm
pandas
mlflow
```

Also, install the [EZPZ library](https://github.com/saforem2/ezpz).

### Prepare dataset
Folder `data/` contains code to prepare the dataset according to the following steps:
- Download reference genome for humans and for mice,
- split into regions, one-hot encode,
- save into HDF5 format.

For personalized genomes, VCF files must be available.

### Train model with toy data
The training script `main_ezpz_mlflow.py` uses the EZPZ library to handle the distributed setting and MLflow to perform experiment tracking.

These trainings were performed on ALCF's Polaris HPC.

```
EZPZ="/path/to/ezpz/"

source $EZPZ/src/ezpz/bin/savejobenv
CKPT_DIR=/path/to/ckpt/dir
unset NCCL_COLLNET_ENABLE NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL
launch python3 main_ezpz_mlflow.py --num_warmup_steps 5000 --ckpt-dir $CKPT_DIR --compile-model
```

`launch` is defined in the `savejobenv` script.


# Citation

```
@article{avsec2021nmeth,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, Ziga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwinska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature Methods},
  year={2021}
}
```

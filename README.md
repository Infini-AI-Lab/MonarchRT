# MonarchRT: Efficient Attention for Real-Time Video Generation

<div align="center">

Krish Agarwal<sup>1</sup>, Zhuoming Chen<sup>1</sup>, Cheng Luo, Yongqi Chen<sup>3</sup>, Haizhong Zheng<sup>1</sup>, Xun Huang<sup>3</sup>, Atri Rudra<sup>2</sup>, Beidi Chen<sup>1</sup>

<sup>1</sup> Carnegie Mellon University &nbsp;&nbsp; <sup>2</sup> University at Buffalo &nbsp;&nbsp; <sup>3</sup> Morpheus AI

</div>

<div align="center">
[<a href="https://arxiv.org/abs/2602.12271">Paper</a>] | [<a href="https://infini-ai-lab.github.io/MonarchRT/">Blog</a>]
</div>
<br>

----------

This repository is an implementation of MonarchRT, a method to sparsely parameterize attention maps in video diffusion transformer (DiT) models using *Monarch matrices*, achieving minimal quality degradation. MonarchRT is effective even for real-time video DiTs that employ few-step diffusion  and autoregressive generation. Using our efficient Triton kernel implementation, MonarchRT achieves, for the first time, **true real-time 16 FPS video generation with Self-Forcing on a single RTX 5090** while continuing to match quality with the dense model (even under a smaller training budget).

<p align="center">  
 <img src="figures/regular_vs_tiled_monarch.png" width="65%"/>
</p>
<p align="center">  
 <img src="figures/kernel_vis.png" width="65%"/>
</p>

## Quick Start

### Installation
Create a conda environment and install dependencies:
```
conda create -n monarch_rt python=3.10 -y
conda activate monarch_rt
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

If you see errors like `ModuleNotFoundError: No module named 'pkg_resources'`, you may need to first downgrade `setuptools` and `pip` to be able to build CLIP (following [this comment](https://github.com/openai/CLIP/issues/528#issuecomment-3878775592)):
```
pip install "setuptools>=65.0.0,<81"
pip install "pip==25.0"
```

### Download checkpoints
Download the released Wan2.1-1.3B and Self-Forcing checkpoints. We also provide instructions for training MonarchRT versions of these models below.
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### CLI Inference
Example inference script using the chunk-wise autoregressive Self-Forcing checkpoint:
```
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
Other config files and corresponding checkpoints can be found in the [configs](configs) folder. For example, you can run training-free MonarchRT on Self-Forcing with:
```
python inference.py \
    --config_path configs/self_forcing_monarch_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
The first time you run this, it will likely take some time for Triton to compile over many autotune configs. The compiled kernels will be cached for subsequent runs, so this time will be reduced. However, autotune will still run each time (while Triton can cache the autotune timings, it currently ignores this using kernel pre-hooks), so you will only see latency improvements with MonarchRT if you generate multiple videos at a time.

## Training

### Autoregressive (Self-Forcing)

#### Download text prompts and ODE initialized checkpoint

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
#### MonarchRT Self-Forcing DMD Training
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_monarch_dmd.yaml \
  --logdir logs/self_forcing_monarch_dmd \
  --disable-wandb
```
This will produce a [torch distributed checkpoint]("https://docs.pytorch.org/docs/stable/distributed.checkpoint.html") under `logs/self_forcing_monarch_dmd/checkpoint_model_000600`, which you can convert using:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch logs/self_forcing_monarch_dmd/checkpoint_model_000600 logs/self_forcing_monarch_dmd/model.pt
```
Then you can use `logs/self_forcing_monarch_dmd/model.pt` as the checkpoint path when performing inference.

#### MonarchRT Causal Initialization

The Self-Forcing training algorithm is data-free when using their provided ODE initialization checkpoint. It is possible to acheive even higher quality by introducing MonarchRT earlier in the training pipeline, specifically during the causal initialization. Although [CausVid](https://github.com/tianweiy/CausVid) performs the initialization using the teacher's ODE trajectories, we find that we can achieve similar results by directly training with diffusion loss on videos generated by the teacher,

##### Download training data

```
huggingface-cli download zhengqili/Self-Forcing-Data --repo-type dataset --include "wanx_14B_shift-3.0_cfg-5.0_lmdb_70K/**" --local-dir data/wanx_14B_shift-3.0_cfg-5.0_lmdb_70K --local-dir-use-symlinks False
```

##### Training

Start with the causal initialization:
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/wan_monarch_causal_training.yaml \
  --logdir logs/wan_monarch_causal \
  --disable-wandb
```
Then convert the produced distributed checkpoint:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch logs/wan_monarch_causal/checkpoint_model_001000 logs/wan_monarch_causal/model.pt
```
Then perform the Self-Forcing DMD training:
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_monarch_from_monarch_dmd.yaml \
  --logdir logs/self_forcing_monarch_from_monarch_dmd \
  --disable-wandb
```
Then convert this distributed checkpoint:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch logs/self_forcing_monarch_from_monarch_dmd/checkpoint_model_000600 logs/self_forcing_monarch_from_monarch_dmd/model.pt
```
Now you can use `logs/self_forcing_monarch_from_monarch_dmd/model.pt` as the checkpoint path when performing inference.

### Bidirectional (Wan 2.1)

MonarchRT is also compatible with traditional bidirectional models. Below are the instructions to train a MonarchRT version of Wan2.1-T2V-1.3B.

#### Download training data

This is the same data used for [causal initialization](#monarchrt-causal-initialization).
```
huggingface-cli download zhengqili/Self-Forcing-Data --repo-type dataset --include "wanx_14B_shift-3.0_cfg-5.0_lmdb_70K/**" --local-dir data/wanx_14B_shift-3.0_cfg-5.0_lmdb_70K --local-dir-use-symlinks False
```

#### Training
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/wan_monarch_finetuning.yaml \
  --logdir logs/wan_monarch_finetuning \
  --disable-wandb
```
Then convert the produced distributed checkpoint:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch logs/wan_monarch_finetuning/checkpoint_model_001000 logs/wan_monarch_finetuning/model.pt
```
Then you can use `logs/wan_monarch_finetuning/model.pt` as the checkpoint path when performing inference.

#### Few-Step Distillation

You can also train a 4-step MonarchRT version of Wan2.1-T2V-1.3B using DMD (which we show results for in our paper).
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/wan_monarch_fewstep_dmd.yaml \
  --logdir logs/wan_monarch_fewstep_dmd \
  --disable-wandb
```
Then convert the produced distributed checkpoint:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch logs/wan_monarch_fewstep_dmd/checkpoint_model_001000 logs/wan_monarch_fewstep_dmd/model.pt
```
Then you can use `logs/wan_monarch_fewstep_dmd/model.pt` as the checkpoint path when performing inference.

## Acknowledgements

The code here is build on the open-source implementation of [Self Forcing](https://github.com/guandeh17/Self-Forcing), which is itself built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
```
@misc{agarwal2026monarchrtefficientattentionrealtime,
      title={MonarchRT: Efficient Attention for Real-Time Video Generation}, 
      author={Krish Agarwal and Zhuoming Chen and Cheng Luo and Yongqi Chen and Haizhong Zheng and Xun Huang and Atri Rudra and Beidi Chen},
      year={2026},
      eprint={2602.12271},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.12271}, 
}
```

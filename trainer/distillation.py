from contextlib import contextmanager
import copy
import gc
import glob
import logging
import numpy as np
import time
import os
import wandb
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.device_mesh import init_device_mesh
from torchvision.io import write_video

from safetensors.torch import load_file

from model import DMD
from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline, BidirectionalInferencePipeline, BidirectionalDiffusionInferencePipeline
from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, launch_distributed_job, send_object, recv_object
from utils.misc import (
    set_seed,
    merge_dict_list
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")
        
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.device_mesh = init_device_mesh(
            "cuda",
            (self.world_size // local_world_size, local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            ema_model = fsdp_wrap(
                copy.deepcopy(self.model.generator),
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                device_mesh=self.device_mesh,
                cpu_offload=getattr(config, "cpu_offload_all", False),
            ) # requires same exact FSDP config as generator
            self.generator_ema = EMA_FSDP(ema_model, decay=ema_weight)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False) or getattr(config, "cpu_offload_all", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if self.global_rank == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            if os.path.isdir(config.generator_ckpt):
                shard_paths = sorted(glob.glob(os.path.join(config.generator_ckpt, "*.safetensors")))
                if not shard_paths:
                    raise ValueError(
                        f"Checkpoint directory {config.generator_ckpt} contains no .safetensors files."
                    )
                state_dict = {}
                for shard_path in shard_paths:
                    print(f"  Loading shard: {os.path.basename(shard_path)}")
                    shard_state = load_file(shard_path, device="cpu")
                    state_dict.update(shard_state)
                state_dict = {f"model.{k}" : v for k, v in state_dict.items()}
            elif config.generator_ckpt.endswith(".safetensors"):
                state_dict = load_file(config.generator_ckpt, device="cpu")
                state_dict = {f"model.{k}" : v for k, v in state_dict.items()}
            else:
                state_dict = torch.load(config.generator_ckpt, map_location="cpu")

            if getattr(config, "init_ema", False) and "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {rename_param(k): v for k, v in state_dict.items()}
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        inf_pipeline_type = (CausalInferencePipeline if self.causal else BidirectionalInferencePipeline) \
            if hasattr(config, 'denoising_step_list') else (CausalDiffusionInferencePipeline if self.causal else BidirectionalDiffusionInferencePipeline)

        self.val_pipeline = inf_pipeline_type(config, self.device, generator=self.model.generator, text_encoder=self.model.text_encoder, vae=self.model.vae)
        self.cpu_group = torch.distributed.new_group(list(range(self.world_size)), backend="gloo")
        if config.validation_prompts_file is not None:
            with open(config.validation_prompts_file, "r") as f:
                all_prompts = [line.strip() for line in f.readlines()]
            validate_first_n = config.validate_first_n if hasattr(config, 'validate_first_n') else len(all_prompts)
            all_prompts = all_prompts[:validate_first_n]
            self.val_prompts = all_prompts
        else:
            self.val_prompts = None

        self.frame_shape = list(self.config.image_or_video_shape)[-3:]

    def save(self):
        ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)

        def _get_state_dict():
            opts = StateDictOptions(full_state_dict=False)
            gen_state = get_model_state_dict(self.model.generator, options=opts)
            critic_state = get_model_state_dict(self.model.fake_score, options=opts)
            state = {"generator": gen_state, "critic": critic_state}
            if self.generator_ema is not None:
                state["generator_ema"] = get_model_state_dict(self.generator_ema.ema_model, options=opts)
            return state

        sharding_strategy = getattr(self.model.generator, "sharding_strategy", None)
        is_hybrid = sharding_strategy == ShardingStrategy.HYBRID_SHARD

        if is_hybrid:
            shard_pg = self.device_mesh["shard"].get_group()
            shard_ranks = dist.get_process_group_ranks(shard_pg)
            is_primary_shard_group = 0 in shard_ranks
            if not is_primary_shard_group:
                return
            state = _get_state_dict()
            dist_cp.save(
                state_dict=state,
                checkpoint_id=ckpt_dir,
                process_group=shard_pg,
            )
        else:
            state = _get_state_dict()
            dist_cp.save(state_dict=state, checkpoint_id=ckpt_dir)

        if self.is_main_process:
            print(f"[DCP] Saved sharded generator checkpoint to {ckpt_dir}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encoder(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, *self.frame_shape],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, *self.frame_shape],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    @contextmanager
    def use_generator_ema(self):
        if self.generator_ema is None:
            # EMA not active
            yield False
            return

        backup = self.model.generator
        self.model.generator = self.generator_ema.ema_model

        try:
            yield True
        finally:
            self.model.generator = backup

    def run_validation(self, label="validation_videos", prompts=None, samples=1, upload=True):
        with torch.no_grad():
            self.model.generator.eval()
            prompts = prompts if prompts is not None else self.val_prompts
            bsz_per_gpu = max(1, (len(prompts) + self.world_size - 1) // self.world_size)
            local_prompts = prompts[self.global_rank * bsz_per_gpu: (self.global_rank + 1) * bsz_per_gpu]
            for sample_num in range(samples):
                validation = []
                for prompt in local_prompts:
                    validation.append(
                        self.generate_video(
                            self.val_pipeline,
                            prompts=[prompt],
                        )
                    )
                if len(local_prompts) < bsz_per_gpu:
                    for _ in range(bsz_per_gpu - len(local_prompts)):
                        self.generate_video(
                            self.val_pipeline,
                            prompts=["dummy_prompt"],
                        )
                if self.is_main_process:
                    all_prompts = list(local_prompts)
                    all_videos = list(validation)

                    for rank in range(1, self.world_size):
                        recv_prompts = recv_object(src=rank, world_size=self.world_size, global_rank=self.global_rank, pg=self.cpu_group)
                        recv_videos = recv_object(src=rank, world_size=self.world_size, global_rank=self.global_rank, pg=self.cpu_group)
                        all_prompts.extend(recv_prompts)
                        all_videos.extend(recv_videos)

                    all_videos = np.concatenate(all_videos, axis=0)
                    log_videos = []
                    output_dir = os.path.join(self.output_path, label, f"step_{self.step:06d}")
                    os.makedirs(output_dir, exist_ok=True)
                    for i, prompt in enumerate(all_prompts):
                        filename = os.path.join(output_dir, f"prompt_{i}_sample_{sample_num}.mp4")
                        write_video(filename, all_videos[i], fps=16)
                        if upload:
                            log_videos.append(wandb.Video(filename, caption=prompt))
                    if upload:
                        logs = { f"{label}_sample{sample_num}" if sample_num > 1 else label : log_videos }
                        wandb.log(logs, step=self.step)
                else:
                    send_object(local_prompts, dst=0, world_size=self.world_size, global_rank=self.global_rank, pg=self.cpu_group)
                    send_object(validation, dst=0, world_size=self.world_size, global_rank=self.global_rank, pg=self.cpu_group)

        self.model.generator.train()
        barrier()

    def train(self):
        start_step = self.step

        while self.step <= self.config.num_training_iters:
            if self.step % self.config.validate_iters == 0 and not (self.disable_wandb or self.val_prompts is None):
                self.run_validation()
                if self.generator_ema is not None and self.step > self.config.ema_start_step:
                    with self.use_generator_ema():
                        self.run_validation("validation_videos_ema")

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None and self.step >= self.config.ema_start_step:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Set EMA params once active
            if (self.step == self.config.ema_start_step) and (self.generator_ema is not None):
                self.generator_ema.copy_(self.model.generator)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        torch.distributed.destroy_process_group(self.cpu_group)
        self.cpu_group = None

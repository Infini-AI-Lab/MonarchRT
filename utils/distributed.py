from datetime import timedelta
from functools import partial
import os
import pickle
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType, LocalStateDictConfig
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint

def fsdp_local_state_dict(fsdp_module: torch.nn.Module):
    cfg = LocalStateDictConfig()  # defaults are fine; tweak if needed
    with FSDP.state_dict_type(fsdp_module, StateDictType.LOCAL_STATE_DICT, cfg):
        return fsdp_module.state_dict()

def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, device_mesh=None):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
        device_mesh=device_mesh,
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)

# send_object and recv_object adapted from FastVideo (https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/distributed/parallel_state.py)

def send_object(obj, dst: int, world_size: int, global_rank: int, pg: ProcessGroup) -> None:
    """Send the input object list to the destination rank."""
    """NOTE: `dst` is the local rank of the destination rank."""

    assert dst < world_size, f"Invalid dst rank ({dst})"

    assert dst != global_rank, (
        "Invalid destination rank. Destination rank is the same "
        "as the current rank.")

    # Serialize object to tensor and get the size as well
    object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

    size_tensor = torch.tensor([object_tensor.numel()],
                                dtype=torch.long,
                                device="cpu")

    # Send object size

    torch.distributed.send(size_tensor,
                            dst=dst,
                            group=pg)

    # Send object
    torch.distributed.send(object_tensor,
                            dst=dst,
                            group=pg)

    return None

def recv_object(src: int, world_size: int, global_rank: int, pg: ProcessGroup):
    """Receive the input object list from the source rank."""
    """NOTE: `src` is the local rank of the source rank."""

    assert src < world_size, f"Invalid src rank ({src})"

    assert src != global_rank, (
        "Invalid source rank. Source rank is the same as the current rank.")

    size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

    # Receive object size
    rank_size = torch.distributed.recv(size_tensor,
                                        src=src,
                                        group=pg)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(  # type: ignore[call-overload]
        size_tensor.item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device="cpu")

    rank_object = torch.distributed.recv(object_tensor,
                                            src=src,
                                            group=pg)

    assert rank_object == rank_size, (
        "Received object sender rank does not match the size sender rank.")

    obj = pickle.loads(object_tensor.numpy().tobytes())

    return obj


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = fsdp_module

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for (n, p), (_, ema_p) in zip(
            fsdp_module.named_parameters(),
            self.ema_model.named_parameters()
        ):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)
    
    @torch.no_grad()
    def copy_(self, fsdp_module):
        for (n, p), (_, ema_p) in zip(
            fsdp_module.named_parameters(),
            self.ema_model.named_parameters()
        ):
            ema_p.data.copy_(p.data)

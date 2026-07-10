import torch
import torch.distributed as dist
from typing import Optional, List, Tuple

class QuatfitParallelismManager:
    """
    Manages 3D Parallelism (Tensor, Pipeline, Expert, Data)
    for distributed training of Quatfit models using torch.distributed.
    """
    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
    ):
        self.tp_size = tensor_model_parallel_size
        self.pp_size = pipeline_model_parallel_size
        self.ep_size = expert_model_parallel_size
        
        # Verify sizes match total world size
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
        self.dp_size = self.world_size // (self.tp_size * self.pp_size)
        assert self.dp_size % self.ep_size == 0, "DP size must be divisible by EP size"
        
        # Parallel group holders
        self.tp_group = None
        self.pp_group = None
        self.ep_group = None
        self.dp_group = None

    def initialize_distributed_groups(self):
        """
        Initializes distributed communication groups.
        """
        if not dist.is_initialized():
            # Mock mode or single GPU fallback
            return
            
        # Group calculations
        # TP groups are contiguous ranks within node
        # PP groups span across nodes
        # DP groups are remaining shards
        for pp in range(self.pp_size):
            for dp in range(self.dp_size):
                tp_ranks = [
                    (pp * self.dp_size * self.tp_size) + (dp * self.tp_size) + tp
                    for tp in range(self.tp_size)
                ]
                group = dist.new_group(tp_ranks)
                if self.rank in tp_ranks:
                    self.tp_group = group
                    
        for tp in range(self.tp_size):
            for dp in range(self.dp_size):
                pp_ranks = [
                    (pp * self.dp_size * self.tp_size) + (dp * self.tp_size) + tp
                    for pp in range(self.pp_size)
                ]
                group = dist.new_group(pp_ranks)
                if self.rank in pp_ranks:
                    self.pp_group = group

        for pp in range(self.pp_size):
            for tp in range(self.tp_size):
                dp_ranks = [
                    (pp * self.dp_size * self.tp_size) + (dp * self.tp_size) + tp
                    for dp in range(self.dp_size)
                ]
                group = dist.new_group(dp_ranks)
                if self.rank in dp_ranks:
                    self.dp_group = group
                    
        # Expert Parallel Group (typically EP shares nodes or matches DP group)
        # Here we shard experts across the EP size
        if self.ep_size > 1:
            # Simple division of DP workers into EP workers
            for pp in range(self.pp_size):
                for tp in range(self.tp_size):
                    for ep_id in range(self.dp_size // self.ep_size):
                        ep_ranks = [
                            (pp * self.dp_size * self.tp_size) + ((ep_id * self.ep_size + ep) * self.tp_size) + tp
                            for ep in range(self.ep_size)
                        ]
                        group = dist.new_group(ep_ranks)
                        if self.rank in ep_ranks:
                            self.ep_group = group

    def get_tp_rank(self) -> int:
        return dist.get_rank(self.tp_group) if self.tp_group is not None else 0

    def get_pp_rank(self) -> int:
        return dist.get_rank(self.pp_group) if self.pp_group is not None else 0

    def get_ep_rank(self) -> int:
        return dist.get_rank(self.ep_group) if self.ep_group is not None else 0

    def get_dp_rank(self) -> int:
        return dist.get_rank(self.dp_group) if self.dp_group is not None else 0

    def broadcast_tp(self, tensor: torch.Tensor):
        if self.tp_group is not None:
            dist.broadcast(tensor, src=0, group=self.tp_group)

    def all_reduce_tp(self, tensor: torch.Tensor):
        if self.tp_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tp_group)

    def all_to_all_ep(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """
        All-to-All communication used to route tokens to different experts
        on different nodes/GPUs during Expert Parallelism.
        """
        if self.ep_group is not None:
            dist.all_to_all_single(output_tensor, input_tensor, group=self.ep_group)
        else:
            output_tensor.copy_(input_tensor)

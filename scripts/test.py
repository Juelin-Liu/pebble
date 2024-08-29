import dataclasses
import os
import torch.distributed as dist
import socket

@dataclasses.dataclass
class DDPMeta:
    local_rank : int
    group_rank : int
    rank: int
    local_world_size : int
    world_size: int
    def __init__(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.group_rank = int(os.environ["GROUP_RANK"])
        self.rank = int(os.environ["RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.world_size = int(os.environ["WORLD_SIZE"])
    
    def update(self):
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["GROUP_RANK"] = str(self.group_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        
    def print(self):
        print("hostname:", socket.gethostname())
        print(f"local_rank={self.local_rank}")
        print(f"group_rank={self.group_rank}")
        print(f"rank={self.rank}")
        print(f"local_world_size={self.local_world_size}")
        print(f"world_size={self.world_size}\n", flush=True)

if __name__ == "__main__":
    print("start testing on:", socket.gethostname(), flush=True)
    ddp_meta = DDPMeta()
    dist.init_process_group(backend="nccl")    
    for i in range(ddp_meta.world_size):
        if i == ddp_meta.rank:
            ddp_meta.print()
            dist.barrier()
    dist.destroy_process_group()

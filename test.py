import os
import torch
import torch.distributed as dist

def ddp_setup(backend="nccl"):
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    # dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])
    dist.init_process_group(backend=backend)
    print(f"ddp setup on G{group_rank}:L{local_rank}")
    # torch.cuda.set_device(local_rank)

def ddp_exit():
    local_rank = int(os.environ["LOCAL_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])
    print(f"ddp exit on G{group_rank}:L{local_rank}")
    dist.destroy_process_group()

if __name__ == "__main__":
    ddp_setup()
    ddp_exit()
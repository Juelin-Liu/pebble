import dgl
import torch
import dataclasses
import random
import json
from tqdm import tqdm
from util import (
    Config,
    Dataset,
    Timer,
    load_dataset,
    get_num_cores,
    get_args,
    get_list_cores,
    get_minibatch_meta,
)
from typing import List


def cache_by_in_degree(data: Dataset) -> torch.Tensor:
    degree = data.graph.in_degrees()
    _, cache_priority = torch.sort(degree, descending=True)
    return cache_priority


def cache_by_out_degree(data: Dataset) -> torch.Tensor:
    degree = data.graph.out_degrees()
    _, cache_priority = torch.sort(degree, descending=True)
    return cache_priority


def cache_by_sampling(data: Dataset) -> torch.Tensor:
    num_epoch = 3
    sampler = dgl.dataloading.NeighborSampler(fanouts=[10, 25])
    dataloader = dgl.dataloading.DataLoader(
        graph=data.graph,
        indices=data.train_mask,
        graph_sampler=sampler,
        batch_size=1024,
        num_workers=get_num_cores(),
    )

    sampling_freq = torch.zeros(data.graph.num_nodes(), dtype=torch.int32)
    with dataloader.enable_cpu_affinity(loader_cores=get_list_cores(0), verbose=False):
        for i in range(num_epoch):
            for input_nodes, _, _ in tqdm(dataloader):
                sampling_freq[input_nodes] += 1

    _, cache_priority = torch.sort(sampling_freq, descending=True)
    return cache_priority


def cache_by_random(data: Dataset) -> torch.Tensor:
    return torch.randperm(data.graph.num_nodes())


@dataclasses.dataclass
class CacheInstance:
    cache_percentage: int = 0
    num_cached: int = 0
    num_partition: int = 1
    cache_mask: torch.Tensor = None
    cache_priority: torch.Tensor = None

    # assume cache is duplicated
    num_hit: torch.Tensor = None  # assume cache is duplicate across all devices
    num_miss: torch.Tensor = None  # assume cache is duplicate across all devices

    # assume cache is partitioned
    num_p2p: torch.Tensor = None  # assume p2p access is enabled
    num_loc_hit: torch.Tensor = None  # assume p2p access is disabled
    num_loc_miss: torch.Tensor = None  # assume p2p access is disabled
    partition_mask: torch.Tensor = None

    def __init__(
        self,
        cache_percentage: int,
        cache_priority: torch.Tensor,
        partition_mask: torch.Tensor,
    ):
        num_nodes = cache_priority.shape[0]
        num_cached = int(num_nodes * cache_percentage / 100)
        num_partition = torch.max(partition_mask).item() + 1
        self.cache_percentage = cache_percentage
        self.num_cached = num_cached
        self.num_partition = num_partition
        self.cache_priority = cache_priority
        self.partition_mask = partition_mask

        self.num_hit = torch.tensor(0)
        self.num_miss = torch.tensor(0)

        self.num_p2p = torch.zeros(
            (self.num_partition, self.num_partition), dtype=torch.int64
        )
        self.num_loc_hit = torch.zeros(self.num_partition, dtype=torch.int64)
        self.num_loc_miss = torch.zeros(self.num_partition, dtype=torch.int64)

        cache_mask = torch.zeros(num_nodes, dtype=torch.bool)
        cache_idx = self.cache_priority[:num_cached].clone()
        cache_mask[cache_idx] = True
        self.cache_mask = cache_mask

    def update_duplicate(self, input_nodes: torch.Tensor):
        hit_mask = self.cache_mask[input_nodes]
        hit = torch.sum(hit_mask)
        miss = input_nodes.shape[0] - hit

        self.num_hit += hit
        self.num_miss += miss

    def update_p2p(self, device_pid: int, input_nodes: torch.Tensor):
        input_pid = self.partition_mask[input_nodes]
        for pid in range(0, self.num_partition):
            pid_id = input_nodes[input_pid == pid]  # input node ids with same pid
            num_hit = torch.sum(self.cache_mask[pid_id])
            self.num_p2p[device_pid][pid] += num_hit

    # This basically gets the diagonal of num_p2p
    def update_loc(self, device_pid: int, input_nodes: torch.Tensor):
        input_pid = self.partition_mask[input_nodes]
        pid_id = input_nodes[input_pid == device_pid]  # input node ids with same pid
        num_hit = torch.sum(self.cache_mask[pid_id])
        num_miss = input_nodes.shape[0] - num_hit
        self.num_loc_hit[device_pid] += num_hit
        self.num_loc_miss[device_pid] += num_miss

    def update(self, input_nodes: torch.Tensor):
        device_pid = random.randint(
            0, self.num_partition - 1
        )  # simulate: randomly selects a device to sample the mini batch
        self.update_duplicate(input_nodes)
        self.update_loc(device_pid, input_nodes)
        self.update_p2p(device_pid, input_nodes)

    def dict(self):
        ret = dict()
        ret["cache_percentage"] = self.cache_percentage
        ret["num_cached"] = self.num_cached
        ret["num_hit"] = self.num_hit.item()
        ret["num_miss"] = self.num_miss.item()
        ret["num_p2p"] = self.num_p2p.tolist()
        ret["num_loc_hit"] = self.num_loc_hit.tolist()
        ret["num_loc_miss"] = self.num_loc_miss.tolist()
        return ret


@dataclasses.dataclass
class CacheManager:
    data: Dataset
    cache_policy: str
    cache_percentage_step: int = 10
    cache_priority: torch.Tensor = None
    cache_instances: List[CacheInstance] = None
    partition_mask: torch.Tensor = None

    def __init__(self, cache_policy: str, data: Dataset, partition_mask: torch.Tensor):
        self.cache_policy = cache_policy
        self.data = data
        self.partition_mask = partition_mask

        assert self.cache_policy in ["in_degree", "out_degree", "random", "sampling"]
        assert self.data.graph is not None

        if self.cache_policy == "in_degree":
            self.cache_priority = cache_by_in_degree(data)
        elif self.cache_policy == "out_degree":
            self.cache_priority = cache_by_out_degree(data)
        elif self.cache_policy == "random":
            self.cache_priority = cache_by_random(data)
        elif self.cache_policy == "sampling":
            self.cache_priority = cache_by_sampling(data)

        self.cache_instances = []
        cache_percentage = 0
        while cache_percentage <= 100:
            self.cache_instances.append(
                CacheInstance(
                    cache_percentage, self.cache_priority, self.partition_mask
                )
            )
            cache_percentage += self.cache_percentage_step

    def update(self, input_nodes: torch.Tensor):
        for inst in self.cache_instances:
            inst.update(input_nodes)

    def dict(self):
        ret = dict()
        ret["cache_policy"] = self.cache_policy
        instances = []
        for inst in self.cache_instances:
            instances.append(inst.dict())

        ret["results"] = instances
        return ret


@dataclasses.dataclass
class CacheSimulator:
    cache_managers: List[CacheManager] = None

    def __init__(self, data: Dataset, partition_mask: torch.Tensor):
        self.cache_managers = []
        for policy in ["in_degree", "out_degree", "random", "sampling"]:
            simu = CacheManager(policy, data, partition_mask)
            self.cache_managers.append(simu)

    def update(self, input_nodes: torch.Tensor):
        for simu in self.cache_managers:
            simu.update(input_nodes)

    # def dict(self):
    #     ret = dict()
    #     for simu in self.cache_managers:
    #         ret[simu.cache_policy] = simu.dict()

    #     return ret

    def list(self):
        ret = []
        for simu in self.cache_managers:
            ret.append(simu.dict())

        return ret


def simulate_cache(config: Config, data: Dataset) -> CacheSimulator:
    sampler = dgl.dataloading.NeighborSampler(fanouts=config.fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph=data.graph,
        indices=data.train_mask,
        graph_sampler=sampler,
        batch_size=config.batch_size,
        num_workers=get_num_cores(),
    )

    partition_mask = torch.randint(
        0, config.num_partition, (data.graph.num_nodes(),), dtype=torch.int8
    )
    simulator = CacheSimulator(data, partition_mask)
    timer = Timer()

    with dataloader.enable_cpu_affinity(loader_cores=get_list_cores(0), verbose=False):
        for epoch in range(config.num_epoch):
            timer.start()

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                simulator.update(input_nodes)

            duration = timer.stop()
            print("simulate epoch {:3d} in {:.2f} secs".format(epoch, duration))

    return simulator


def log_simulation(config: Config, data: Dataset, simu: CacheSimulator):
    assert config.log_file.endswith(".json")
    print("writing log file to", config.log_file)
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_minibatch_meta(config, data))
        ret["results"] = simu.list()
        json.dump(ret, outfile)


def main():
    config = get_args()
    data = load_dataset(config, True)
    simu = simulate_cache(config, data)
    log_simulation(config, data, simu)


if __name__ == "__main__":
    main()

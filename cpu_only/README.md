# GNN Experiments using CPU only

## Run Simulation on MiniBatch Training
The scripts simulate the amount of feature data loaded from the GPU cache or through PCIe buses.

It has implemented 4 caching strategies:
1. out-degree: cache nodes with higher out-degrees first.
2. in-degree: cache nodes with higher in-degrees first.
3. sampling: sample the graph using fanouts [10,25] for 3 epochs and cache the nodes with higher frequencies first.
4. random: randomly selects nodes to be cached in gpu.

The simulator will profile cache hit/miss at different cache percentage from 0% - 100% at a step of 10%.

You can change the number of devices used (--num_partitions) in the simulation the amount of cross device communication.

In this implementation, the partition ids are generated randomly which means the cached data will be uniformly (and randomly) stored on each device.

To run mini-batch simulation for dataloading workload, the `log_path` must end with `.json`.

Example, run mini batch graph training with batch size 1024 and fanouts [15,15], assuming 4 partitions (devices) are used:
```bash
python3 simulate_minibatch_cache.py --data_dir DATA_DIRECTORY --graph_name ogbn-arxiv --batch_size 1024 --fanouts 15,15 --num_partition=4 --log_path log.json
```

### Output Format Version 1
The JSON output has the following schema:

```json
{
    "version": ...,
    "graph_name": ...,
    "num_node": ...,
    "feat_width": ...,
    "fanouts": ...,
    "num_epoch": ...,
    "num_partition": ...,
    "results": [
        // Results for different cache policy
        {
            "cache_policy": ...,
            "results": [
                // Results for different cache percentage
                {
                    "cache_percentage": ..., // percentage of feaure cached
                    "num_cache": ..., // number of feature cached

                    // assuming all cached data is duplicated on all devices (all cached data is accessible)
                    "num_hit": ..., // total number of hits
                    "num_miss": ..., // total number of misses

                    // assuming cached data are partitioned across devices and all cached data is accessible (nvlink)
                    "num_p2p": ..., // num_p2p[x][y]: number of times y cached x's required data (dataflow: y -> x)
                    
                    // assuming cached data are partitioned across devices and only local cached data is accessible (no-nvlink)
                    "num_loc_hit": ..., // num_p2p[x]: number of times x cached the required data
                    "num_loc_miss": ..., // num_p2p[x]: number of times x does not cache the required data
                }, ...
            ]
        }, ...
    ]
}
```

## Full Graph Training

Example, run full graph training with evaluation:
```bash
python3 train_full.py --data_dir DATA_DIRECTORY --graph_name ogbn-arxiv --eval
```


## MiniBatch Training

Example, run minibatch training with batch size 1024 and fanouts [15, 15] without evaluation:
```bash
python3 train_minibatch.py --data_dir DATA_DIRECTORY --graph_name ogbn-arxiv --batch_size 1024 --fanouts 15,15
```
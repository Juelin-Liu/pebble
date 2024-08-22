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

The `log_path` must end with `.json`.

Example, run mini batch graph training with batch size 1024 and fanouts [15,15], assuming 4 partitions (devices) are used:
```bash
python3 simulate_minibatch_dataload.py --graph_name ogbn-arxiv --batch_size 1024 --fanouts 15,15 --num_partition=4 --log_path log.json --data_dir YOUR_DATASET_DIR
```

### Simulation Output Format Version 1
The JSON output has the following schema:

```yaml
{
    "version": 1,
    "graph_name": "ogbn-arxiv",
    "num_node": 169343,
    "feat_width": 128,
    "fanouts": [15, 15, 15],
    "num_epoch": 1,
    "num_partition": 4,
    "results": [
        // Results for different cache policy
        {
            // Cache policy determines the feature of which nodes have priority to be cached in GPU memory
            // In this case, feature of nodes with higher in-degree will be cached first
            // We can modify this in the python code to enable simulating more caching strategies
            "cache_policy": "in_degree",
            "results": [
                // Results for different cache percentage
                {
                    // main configuration (the amount of data is cached in GPU memory, the rest is on CPU memory)
                    "cache_percentage": 0, // percentage of feature that is cached in GPU memory
                    "num_cache": 0, // number of feature cached


                    // Simulate data transfer during minibatch training
                    // 1. Cached data is duplicated on all GPUs
                    "num_hit": 0, // total number of hits
                    "num_miss": 3151979, // total number of misses

                    // 2. Cached data is partitioned among all GPUs, which can fetch data from each other
                    // num_p2p is a 2D array with shape (num_partition x num_partition)
                    // num_p2p[a, b] means number of features that are transfered from b to a
                    "num_p2p": [[...],
                                [...],
                                [...],
                                ...] 
                    
                    // 3. Cached data is partitioned among all GPUs, however they cannot fetch data from each other
                    // This is useful if graph topology is also partitioned during minibatch training and each gpu can only access its local graph and feature data.
                    // Results are 1D arrays of shape (num_partition)
                    // num_p2p[a]: number of times partition a cached the required data
                    "num_loc_hit": [...], 

                    // num_p2p[a]: number of times partition a does not cache the required data
                    "num_loc_miss": [...] 

                }, 
                // other results with different cache rate
            ]
        }, 
        // other results with different cache policy
    ]
}
```

## Full Graph Training
Example, run full graph training:
```bash
python3 train_full.py --graph_name ogbn-arxiv --epoch_num 10 --log_file log.json --data_dir YOUR_DATASET_DIR
```

### Full Graph Training Output Format Version 1

```yaml
{
    "version": 1,
    "graph_name": "ogbn-arxiv",
    "mode": "full",
    "num_node": 169343,
    "feat_width": 128,
    "num_epoch": 10,
    "num_partition": 1,
    "weight_decay": 0.0005,
    "learning_rate": 0.005,
    "test_acc": 0.31954817603851615,
    "results": [
        {
            "epoch": 0,
            "eval_acc": 0.07627772744051814,
            "forward_time": 0.5917582511901855,
            "backward_time": 0.890723466873169,
            "cur_epoch_time": 1.4824936389923096,
            "acc_epoch_time": 1.4824936389923096,
            "evaluate_time": 0.5268638134002686,
            "loss": 3.8400356769561768
        },
        {
            "epoch": 1,
            "eval_acc": 0.07634484378670425,
            "forward_time": 0.5696797370910645,
            "backward_time": 0.7793552875518799,
            "cur_epoch_time": 1.3490421772003174,
            "acc_epoch_time": 2.831535816192627,
            "evaluate_time": 0.5136260986328125,
            "loss": 3.2898571491241455
        },
        // more epoch's results
    ]
}
```

## MiniBatch Training

Example, run minibatch training with batch size 1024 and fanouts [15, 15, 15]:
```bash
python3 train_minibatch.py --graph_name ogbn-arxiv --batch_size 1024 --fanouts 15,15,15 --data_dir YOUR_DATASET_DIR
```

### MiniBatch Training Output Format Version 1

```yaml
{
    "version": 1,
    "graph_name": "ogbn-arxiv",
    "mode": "minibatch",
    "num_node": 169343,
    "feat_width": 128,
    "batch_size": 1024,
    "fanouts": [15, 15, 15],
    "num_epoch": 10,
    "num_partition": 1,
    "weight_decay": 0.0005,
    "learning_rate": 0.005,
    "test_acc": 0.5390819311141968,
    "results": [
        {
            "epoch": 0,
            "eval_acc": 0.5471971035003662,
            "sample_time": 1.536076307296753,
            "load_time": 0.16553139686584473,
            "forward_time": 2.787198305130005,
            "backward_time": 4.274858713150024,
            "cur_epoch_time": 8.763873100280762,
            "acc_epoch_time": 8.763873100280762,
            "evaluate_time": 4.784973382949829,
            "loss": 1.6351159811019897
        },
        {
            "epoch": 1,
            "eval_acc": 0.573197603225708,
            "sample_time": 1.3836567401885986,
            "load_time": 0.19238638877868652,
            "forward_time": 2.2498533725738525,
            "backward_time": 3.8861567974090576,
            "cur_epoch_time": 7.712284803390503,
            "acc_epoch_time": 16.476157903671265,
            "evaluate_time": 4.910733461380005,
            "loss": 1.4281635284423828
        }, 
        // more epoch's results
    ]
}
```
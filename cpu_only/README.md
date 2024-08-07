# GNN Benchmarks using CPU only


## Full Graph Training

```bash
python3 train_full.py --data_dir DATA_DIRECTORY --graph_name ogbn-arxiv --eval
```


## MiniBatch Training

```bash
python3 train_minibatch.py --data_dir DATA_DIRECTORY --graph_name ogbn-arxiv --batch_size 1024 --fanouts 15,15
```
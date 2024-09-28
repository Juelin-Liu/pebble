#!/bin/bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu124/repo.html
conda install -c dglteam/label/th24_cu124 dgl

pip install ninja cmake ogb torchmetrics pandas

git clone -b cuda-12.4 https://github.com/Juelin-Liu/torch-quiver.git 

cd torch-quiver
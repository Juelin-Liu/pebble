#!/bin/bash

aria2c -s 16 -x 16 http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip

aria2c -s 16 -x 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip

aria2c -s 16 -x 16 http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip

aria2c -s 128 -x 16 http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip



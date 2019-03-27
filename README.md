# Tensor Product Representation Recurrent Neural Network
This repository containes PyTorch implementation of
paper [*Learning to Reason with Third-Order Tensor Products*](https://papers.nips.cc/paper/8203-learning-to-reason-with-third-order-tensor-products)
published at NeurIPS, 2018.
TPR-RNN is applied to the bAbI tasks and achieves SOTA results.
This implementation is primarily based on the [*original
implementation*](https://github.com/ischlag/TPR-RNN).

# Requirements
- Python 3.6
- Pytorch==1.0.0
- tensorboardX==1.5

## How to setup environment
1. [Download and install conda](https://conda.io/docs/user-guide/install/download.html)
2. Create conda environment from environment.yml file
```
conda env create -n tpr_rnn -f environment.yml
```
3. Activate conda environment
```
source activate tpr_rnn
```

# Usage
Run the pre-trained model.
```bash
python3 eval.py --model-dir PATH [--no-cuda]
```

Train from scratch. (Look at the train.py files for details)
```bash
python3 train.py --config-file PATH --serialization-path PATH
[--eval-test] [--logging-level LEVEL]
```

Cluster analysis
```bash
python3 cluster_analysis.py --model-path PATH [--num-stories N]
```

# TODO
- [x] Pre-train on all tasks
- [x] Cluster analysis
- [ ] Task transformation

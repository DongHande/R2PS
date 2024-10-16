# R2PS: Retriever and Ranker Framework with Probabilistic Hard Negative Sampling for Code Search

This is the official pytorch implementation of R2PS, a well-performed code search method for recommendation system. R2PS is proposed in the paper:

[Retriever and Ranker Framework with Probabilistic Hard Negative Sampling for Code Search](https://arxiv.org/abs/2305.04508)

by Hande Dong, Jiayi Lin, Yichong Leng, Jiawei Chen, and Yutao Xie. 

Published at APSEC 2024. 

## Introduction

P2PS is a retriever-ranker based code search method, with probabilistic hard negative sampling to train better ranker model. This method is valid to improve the performance of code search according to our comperhensive experiments. 

## Required Envirnment

```
torch
transformers
argparse
logging
random
numpy
```

## Run the Code

```shell
python run_unixcoder_rr_hard.py
```

## Steps to Reproduce Experiment Results

1. **Download and prepare dataset**: For example, we can download and prepare [CSN](https://github.com/github/CodeSearchNet) according to their instruction. 
2. **Train a dual-encoder model**: To train a dual-encoder model for code search, we can refer to [the official code of UniXcoder](https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/downstream-tasks/code-search). 
3. **Train a cross-encoder**: By running "python run_unixcoder_rr_hard.py" command, we can train a cross-encoder with hard negative sampling method according to a well-trained dual-encoder. 
4. **Inference with retrieve and ranker framework**: relevant code is included in the run_unixcoder_rr_hard.py script. 


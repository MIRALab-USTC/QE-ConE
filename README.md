# ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs

This is the code of paper 
**ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs**. 
Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, Feng Wu. NeurIPS 2021. [[arXiv](https://arxiv.org/abs/2110.13715)]

## Requirements
- Python 3.7
- PyTorch 1.7
- tqdm


## Reproduce the Results
1. Download the datasets [here](http://snap.stanford.edu/betae/KG_data.zip).
2. Move the zipped datasets to the root directory of ConE and run `unzip -d data KG_data.zip`.
3. Run the scripts in `scripts.sh`.


## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{NEURIPS2021_QECONE,
 author = {Zhanqiu Zhang and Jie Wang and Jiajun Chen and Shuiwang Ji and Feng Wu},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs},
 year = {2021}
}
```

## Acknowledgement
We refer to the code of [KGReasoning](https://github.com/snap-stanford/KGReasoning). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following papers useful.

**Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion.**
*Zhanqiu Zhang, Jianyu Cai, Jie Wang.* NeurIPS 2020. [[paper](https://arxiv.org/abs/2011.05816)] [[code](https://github.com/MIRALab-USTC/KGE-DURA)]

**Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction.**
*Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang.* AAAI 2020. [[paper](https://arxiv.org/pdf/1911.09419.pdf)] [[code](https://github.com/MIRALab-USTC/KGE-HAKE)]

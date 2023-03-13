# TO DO: Clustering CNG 

- --how_to_init_mu_sub=kmeans有bug，还没解决
- 现在多张卡跑会奇怪的在某个epoch卡住，有时候就没事。怀疑跟slurm调度有关
- 多张卡跑wandb的logging会重复n次，因为pl的wandblogger在offline的时候不上传，而slurm计算节点不连网，因此采用再最开始init。而pl.Trainer也许是fork还是怎么搞的，反正wandb会当成有很多个run在跑


SCAN 分类, MoCo

no_split_and_merge init_k=5 

看分类结果

从20->几个重新extract features


深究贝叶斯




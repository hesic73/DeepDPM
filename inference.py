import torch
from torch.utils.data import DataLoader
from src.clusternet_models.clusternetasmodel import ClusterNetModel
from argparse import Namespace
from src.datasets import CustomDataset
from tqdm import tqdm
import numpy as np

checkpoint_path = "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/DeepDPM/saved_models/CNG/init_k_5/epoch=499-step=39499.ckpt"
data_dim = 128
cp_state = torch.load(checkpoint_path)
K = cp_state['state_dict']['cluster_net.class_fc2.weight'].shape[0]
hyper_param = cp_state['hyper_parameters']

args = Namespace()
for key, value in hyper_param.items():
    setattr(args, key, value)

args.batch_size = 512

model = ClusterNetModel.load_from_checkpoint(checkpoint_path,
                                             input_dim=data_dim,
                                             init_k=K,
                                             hparams=args)


model.eval()
model.cuda()
dataset_obj = CustomDataset(args)
train_loader = DataLoader(dataset_obj.get_train_data(), batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers,)
cluster_assignments = []
for data, label in tqdm(train_loader):
    soft_assign = model(data.cuda())
    hard_assign = soft_assign.argmax(-1)
    cluster_assignments.append(hard_assign.cpu())

cluster_assignments = torch.concat(cluster_assignments)

unique_ids, cnts = cluster_assignments.unique(
    return_counts=True)

print(unique_ids)
print(cnts)

clusters = {}
for id in unique_ids:
    tmp = torch.where(cluster_assignments == id)[0]
    clusters[id] = tmp
    
np.save("results/clusters.npy",clusters)

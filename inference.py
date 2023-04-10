import torch
from torch import Tensor
from torch.utils.data import DataLoader
from src.clusternet_models.clusternetasmodel import ClusterNetModel
from argparse import Namespace
from src.datasets import CustomDataset
from tqdm import tqdm
import numpy as np
device = 'cuda'


checkpoint_path = "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/DeepDPM/saved_models/proteasome-12/test7_D16_nu300/epoch=499-step=447426.ckpt"
data_dim = 128
cp_state = torch.load(checkpoint_path, map_location=device)
K = cp_state['state_dict']['cluster_net.class_fc2.weight'].shape[0]
hyper_param = cp_state['hyper_parameters']['hparams']

hyper_param.batch_size = 512

model = ClusterNetModel.load_from_checkpoint(checkpoint_path,
                                             input_dim=data_dim,
                                             init_k=K,
                                             hparams=hyper_param)


model.eval()
model.to(device)
dataset_obj = CustomDataset(hyper_param)
train_loader = DataLoader(
    dataset_obj.get_train_data(),
    batch_size=hyper_param.batch_size,
    shuffle=False,
    num_workers=hyper_param.num_workers,
)
cluster_assignments = []
cluster_probabilities = []
with torch.no_grad():
    for data, label in tqdm(train_loader):
        soft_assign: Tensor = model(data.to(device))

        p, hard_assign = torch.max(soft_assign, dim=-1)
        cluster_assignments.append(hard_assign.cpu())
        cluster_probabilities.append(p.cpu())

cluster_assignments = torch.concat(cluster_assignments)
cluster_probabilities = torch.concat(cluster_probabilities)

unique_ids, cnts = cluster_assignments.unique(
    return_counts=True)

print(unique_ids)
print(cnts)

clusters = {}
for id in unique_ids:
    tmp = torch.where(cluster_assignments == id)[0]
    tmp_p = cluster_probabilities[tmp]
    tmp_p, indices = torch.sort(tmp_p, descending=True)
    print(tmp_p[:10])
    tmp = tmp[indices]
    clusters[id] = tmp

np.save("results/proteasome12_clusters.npy", clusters)

from torch import Tensor
import torch
from sklearn.cluster import KMeans
from typing import Tuple


def GPU_KMeans(X: Tensor,
               num_clusters: int,
               device=torch.device('cpu'),
               tqdm_flag: bool = False) -> Tuple[Tensor, Tensor]:
    """In replace of original GPU_KMeans, as currently numba isn't available in python 3.11.
    By the way, according to the comparision in the repository, GPU-version kmeans doesn't show
    a boost in performance compared to sklearn.cluster.KMeans

    Args:
        X (Tensor): (n,codes_dim)
        num_clusters (int): #clusters
        device (_type_, optional): placeholder. Defaults to torch.device('cpu').
        tqdm_flag (bool, optional): placeholder. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: labels (n,) ; kmeans_mus (num_clusters,codes_dim)
    """

    X = X.cpu()
    # assert torch.device(device) == torch.device('cpu')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(X)
    labels = torch.from_numpy(kmeans.labels_)
    kmeans_mus = torch.from_numpy(kmeans.cluster_centers_)

    return labels, kmeans_mus

#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP_Classifier(nn.Module):
    def __init__(
        self,
        hparams,
        codes_dim: int = 320,
        k: Optional[int] = None,
    ):
        super(MLP_Classifier, self).__init__()

        self.k: int = hparams.init_k if k is None else k

        self.codes_dim: int = codes_dim
        self.hidden_dims: int = hparams.clusternet_hidden_layer_list
        self.last_dim: int = self.hidden_dims[-1]
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dims[0])
        hidden_modules = []
        for i in range(len(self.hidden_dims) - 1):
            hidden_modules.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            hidden_modules.append(nn.BatchNorm1d(self.hidden_dims[i + 1]))
            hidden_modules.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_modules)
        self.class_fc2 = nn.Linear(self.hidden_dims[-1], self.k)
        # print(self.hidden_layers)

        self.softmax_norm = hparams.softmax_norm

    def forward(self, x):
        x = x.view(-1, self.codes_dim)
        x = F.relu(self.class_fc1(x))
        x = self.hidden_layers(x)
        x = self.class_fc2(x)
        x = torch.mul(x, self.softmax_norm)
        return F.softmax(x, dim=1)

    def update_K_split(self,
                       split_decisions,
                       init_new_weights="same",
                       subclusters_nets=None):
        # split_decisions is a list of K booleans indicating whether to split a cluster or not
        # update the classifier to have K' more classes, where K' is the number of clusters that should be split
        # deleting the old clustering and addind the new ones to the end (weights)

        class_fc2 = self.class_fc2
        mus_ind_to_split = torch.nonzero(split_decisions, as_tuple=False)
        self.k += len(mus_ind_to_split)

        with torch.no_grad():
            self.class_fc2 = nn.Linear(self.last_dim, self.k)

            # Adjust weights
            weights_not_split = class_fc2.weight.data[
                torch.logical_not(split_decisions.bool()), :]
            weights_splits = class_fc2.weight.data[split_decisions.bool(), :]
            new_weights = self._initalize_weights_split(
                weights_splits,
                split_decisions,
                init_new_weight=init_new_weights,
                subclusters_nets=subclusters_nets)

            self.class_fc2.weight.data = torch.cat(
                [weights_not_split, new_weights])

            # Adjust bias
            bias_not_split = class_fc2.bias.data[torch.logical_not(
                split_decisions.bool())]
            bias_split = class_fc2.bias.data[split_decisions.bool()]
            new_bias = self._initalize_bias_split(
                bias_split,
                split_decisions,
                init_new_weight=init_new_weights,
                subclusters_nets=subclusters_nets)
            self.class_fc2.bias.data = torch.cat([bias_not_split, new_bias])

    def update_K_merge(self,
                       merge_decisions,
                       pairs_to_merge,
                       highest_ll,
                       init_new_weights="same"):
        """ Update the clustering net after a merge decision was made

        Args:
            merge_decisions (torch.tensor): a list of K booleans indicating whether to a cluster should be merged or not
            pairs_to_merge ([type]): a list of lists, which list contains the indices of two clusters to merge
            init_new_weights (str, optional): How to initialize the weights of the new weights of the merged cluster. Defaults to "same".
                "same" uses the weights of the cluster with the highest loglikelihood, "random" uses random weights.
            highest_ll ([type]): a list of the indices of the clusters with the highest log likelihood for each pair.

        Description:
            We will delete the weights of the two merged clusters, and append (to the end) the weights of the newly merged clusters
        """

        self.k -= len(highest_ll)

        with torch.no_grad():
            class_fc2 = nn.Linear(self.last_dim, self.k)

            # Adjust weights
            weights_not_merged = self.class_fc2.weight.data[
                torch.logical_not(merge_decisions), :]
            weights_merged = self.class_fc2.weight.data[merge_decisions, :]
            new_weights = self._initalize_weights_merge(
                weights_merged,
                merge_decisions,
                highest_ll,
                init_new_weight=init_new_weights)

            class_fc2.weight.data = torch.cat(
                [weights_not_merged, new_weights])

            # Adjust bias
            bias_not_merged = self.class_fc2.bias.data[torch.logical_not(
                merge_decisions)]
            bias_merged = self.class_fc2.bias.data[merge_decisions]
            new_bias = self._initalize_bias_merge(
                bias_merged,
                merge_decisions,
                highest_ll,
                init_new_weight=init_new_weights)
            class_fc2.bias.data = torch.cat([bias_not_merged, new_bias])

            self.class_fc2 = class_fc2

    def _initalize_weights_split(self,
                                 weight: torch.Tensor,
                                 split_decisions,
                                 init_new_weight,
                                 subclusters_nets=None):
        if init_new_weight == "same":
            # just duplicate, can think of something more complex later
            return weight.repeat(1, 2).view(-1, self.last_dim)
        elif init_new_weight == "random":
            return torch.FloatTensor(weight.shape[0] * 2,
                                     weight.shape[1]).uniform_(
                                         -1., 1).to(device=self.device)
        elif init_new_weight == "subclusters":
            new_weights = []
            for k, split in enumerate(split_decisions):
                if split:
                    new_weights.append(subclusters_nets.class_fc2.weight.data[
                        2 * k:2 * (k + 1),
                        self.last_dim * k:self.last_dim * (k + 1)].clone())
            return torch.cat(new_weights)
        else:
            raise NotImplementedError

    def _initalize_weights_merge(self,
                                 weights_merged,
                                 merge_decisions,
                                 highest_ll,
                                 init_new_weight="same"):
        if init_new_weight == "same":
            # Take the weights of the cluster with highest likelihood
            ll = [i[0].item() for i in highest_ll]
            return self.class_fc2.weight.data[ll, :]
        elif init_new_weight == "random":
            return torch.FloatTensor(len(highest_ll),
                                     weights_merged.shape[1]).uniform_(
                                         -1., 1).to(device=self.device)
        elif init_new_weight == "average":
            raise NotImplementedError()
        else:
            raise NotImplementedError

    def _initalize_bias_split(self,
                              bias_split,
                              split_decisions,
                              init_new_weight,
                              subclusters_nets=None):
        if init_new_weight == "same":
            # just duplicate
            return bias_split.repeat_interleave(2)
        elif init_new_weight == "random":
            return torch.zeros(bias_split.shape[0] * 2).to(device=self.device)
        elif init_new_weight == "subclusters":
            new_bias = []
            for k, split in enumerate(split_decisions):
                if split:
                    new_bias.append(
                        subclusters_nets.class_fc2.bias.data[2 * k:2 *
                                                             (k + 1)].clone())
            return torch.cat(new_bias)
        else:
            raise NotImplementedError

    def _initalize_bias_merge(self,
                              bias_merged,
                              merge_decisions,
                              highest_ll,
                              init_new_weight="same"):
        if init_new_weight == "same":
            # take the biases of the highest likelihood
            ll = [i[0].item() for i in highest_ll]
            return self.class_fc2.bias.data[ll]
        elif init_new_weight == "random":
            return torch.zeros(len(highest_ll)).to(device=self.device)
        elif init_new_weight == "average":
            raise NotImplementedError
        else:
            raise NotImplementedError


class Subclustering_net(nn.Module):
    # Duplicate only inner layer
    # SHAPE is input dim -> 50 * K -> 2 * K
    def __init__(self,
                 hparams,
                 codes_dim: int = 320,
                 hidden_dim: int = 50,
                 k: Optional[int] = None):
        super(Subclustering_net, self).__init__()
        if k is None:
            self.K: int = hparams.init_k
        else:
            self.K: int = k

        self.codes_dim: int = codes_dim
        self.hparams = hparams
        self.hidden_dim: int = hidden_dim
        self.softmax_norm = hparams.softmax_norm

        self.device = "cuda" if torch.cuda.is_available(
        ) and hparams.gpus is not None else "cpu"

        # the subclustering net will be a stacked version of the clustering net
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K)
        self.class_fc2 = nn.Linear(self.hidden_dim * self.K, 2 * self.K)

        gradient_mask_fc2 = torch.zeros(2 * self.K, self.hidden_dim * self.K)
        # detach different subclustering nets - zeroing out the weights connecting between different subnets
        # and also zero their gradient
        for k in range(self.K):
            gradient_mask_fc2[2 * k:2 * (k + 1), self.hidden_dim *
                              k:self.hidden_dim * (k + 1)] = 1

        self.class_fc2.weight.data *= gradient_mask_fc2
        self.hood_handle=self.class_fc2.weight.register_hook(
            lambda grad: grad.mul_(gradient_mask_fc2.to(device=self.device)))
        # weights are zero and their grad will always be 0 so won't change

    def forward(self, X: torch.Tensor):
        # X (n,codes_dim)
        # Note that there is no softmax here
        X = F.relu(self.class_fc1(X))
        X = self.class_fc2(X)
        return X

    def update_K_split(self,
                       split_decisions: torch.Tensor,
                       init_new_weights="same"):
        # split_decisions is a list of K booleans indicating whether to split a cluster or not
        # update the classifier to have K' more classes, where K' is the number of clusters that should be split
        # deleting the old clustering and addind the new ones to the end (weights)

        class_fc1 = self.class_fc1
        class_fc2 = self.class_fc2
        mus_ind_to_split = torch.nonzero(split_decisions, as_tuple=False)
        mus_ind_not_split = torch.nonzero(torch.logical_not(split_decisions),
                                          as_tuple=False)
        self.K += len(mus_ind_to_split)

        with torch.no_grad():
            self.class_fc1 = nn.Linear(self.codes_dim,
                                       self.hidden_dim * self.K)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, 2 * self.K)

            # Adjust weights
            fc1_weights_not_split = class_fc1.weight.data[torch.logical_not(
                split_decisions.bool()).repeat_interleave(self.hidden_dim), :]
            fc1_weights_split = class_fc1.weight.data[
                split_decisions.bool().repeat_interleave(self.hidden_dim), :]
            fc1_new_weights = self._initalize_weights_split(
                fc1_weights_split, init_new_weight=init_new_weights)
            self.class_fc1.weight.data = torch.cat(
                [fc1_weights_not_split, fc1_new_weights])
            self.class_fc2.weight.data.fill_(0)
            gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K,
                                            2 * self.K)
            for i, k in enumerate(mus_ind_not_split):
                # i is the new index of the cluster and k is the old one
                self.class_fc2.weight.data[
                    2 * i:2 * (i + 1), self.hidden_dim * i:self.hidden_dim *
                    (i + 1)] = class_fc2.weight.data[2 * k:2 * (k + 1),
                                                     self.hidden_dim *
                                                     k:self.hidden_dim *
                                                     (k + 1)]
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim *
                                  (i + 1), 2 * i:2 * (i + 1)] = 1
            for j, k in enumerate(mus_ind_to_split.repeat_interleave(2)):
                # j + len(mus_ind_not_split) is the new index and k is the old one. We use interleave to create 2 new clusters for each split cluster
                i = j + len(mus_ind_not_split)
                weights = class_fc2.weight.data[2 * k:2 * (k + 1),
                                                self.hidden_dim *
                                                k:self.hidden_dim * (k + 1)]
                if init_new_weights != 'same':
                    weights = self._initalize_weights_split(weights,
                                                            init_new_weights,
                                                            num=1)
                
                print(self.class_fc2.weight.shape)
                self.class_fc2.weight.data[2 * i:2 * (i + 1), self.hidden_dim *
                                           i:self.hidden_dim *
                                           (i + 1)] = weights
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim *
                                  (i + 1), 2 * i:2 * (i + 1)] = 1

            self.class_fc2.weight.register_hook(lambda grad: grad.mul_(
                gradient_mask_fc2.T.to(device=self.device)))

            # Adjust bias
            fc1_bias_not_split = class_fc1.bias.data[torch.logical_not(
                split_decisions.bool()).repeat_interleave(self.hidden_dim)]
            fc1_bias_split = class_fc1.bias.data[
                split_decisions.bool().repeat_interleave(self.hidden_dim)]
            fc2_bias_not_split = class_fc2.bias.data[torch.logical_not(
                split_decisions.bool()).repeat_interleave(2)]
            fc2_bias_split = class_fc2.bias.data[
                split_decisions.bool().repeat_interleave(2)]

            fc1_new_bias = self._initalize_bias_split(
                fc1_bias_split, init_new_weight=init_new_weights)
            fc2_new_bias = self._initalize_bias_split(
                fc2_bias_split, init_new_weight=init_new_weights)
            self.class_fc1.bias.data = torch.cat(
                [fc1_bias_not_split, fc1_new_bias])
            self.class_fc2.bias.data = torch.cat(
                [fc2_bias_not_split, fc2_new_bias])

            self.class_fc1.to(device=self.device)
            self.class_fc2.to(device=self.device)

            del class_fc1, class_fc2

    def update_K_merge(self,
                       merge_decisions,
                       pairs_to_merge,
                       highest_ll,
                       init_new_weights="highest_ll"):
        """ Update the clustering net after a merge decision was made

        Args:
            merge_decisions (torch.tensor): a list of K booleans indicating whether to a cluster should be merged or not
            pairs_to_merge ([type]): a list of lists, which list contains the indices of two clusters to merge
            init_new_weights (str, optional): How to initialize the weights of the new weights of the merged cluster. Defaults to "same".
                "same" uses the weights of the cluster with the highest loglikelihood, "random" uses random weights.
            highest_ll ([type]): a list of the indices of the clusters with the highest log likelihood for each pair.

        Description:
            We will delete the weights of the two merged clusters, and append (to the end) the weights of the newly merged clusters
        """

        class_fc1 = self.class_fc1
        class_fc2 = self.class_fc2
        mus_ind_not_merged = torch.nonzero(torch.logical_not(
            merge_decisions.clone().detach()),
            as_tuple=False)
        self.K -= len(highest_ll)

        with torch.no_grad():
            self.class_fc1 = nn.Linear(self.codes_dim,
                                       self.hidden_dim * self.K)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, 2 * self.K)

            # Adjust weights
            fc1_weights_not_merged = class_fc1.weight.data[
                torch.logical_not(merge_decisions.clone().detach()
                                  ).repeat_interleave(self.hidden_dim), :]
            fc1_new_weights = []
            fc1_new_bias = []
            fc2_new_bias = []
            for merge_pair, highest_ll_k in zip(pairs_to_merge, highest_ll):
                fc1_weights_merged = [
                    class_fc1.weight.data[k * self.hidden_dim:(k + 1) *
                                          self.hidden_dim, :]
                    for k in merge_pair
                ]
                fc1_new_weights.append(
                    self._initalize_weights_merge(
                        fc1_weights_merged,
                        (torch.tensor(highest_ll_k) == merge_pair[1]).item(),
                        init_new_weight=init_new_weights))

                fc1_bias_merged = [
                    class_fc1.bias.data[k * self.hidden_dim:(k + 1) *
                                        self.hidden_dim] for k in merge_pair
                ]
                fc1_new_bias.append(
                    self._initalize_weights_merge(
                        fc1_bias_merged,
                        (torch.tensor(highest_ll_k) == merge_pair[1]).item(),
                        init_new_weight=init_new_weights))
            fc1_new_weights = torch.cat(fc1_new_weights)
            fc1_new_bias = torch.cat(fc1_new_bias)

            self.class_fc1.weight.data = torch.cat(
                [fc1_weights_not_merged, fc1_new_weights])

            self.class_fc2.weight.data.fill_(0)
            gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K,
                                            2 * self.K)
            for i, k in enumerate(mus_ind_not_merged):
                # i is the new index of the cluster and k is the old one
                self.class_fc2.weight.data[2 * i: 2*(i + 1), self.hidden_dim * i: self.hidden_dim * (i+1)] =\
                    class_fc2.weight.data[2 * k: 2 *
                                          (k + 1), self.hidden_dim * k: self.hidden_dim * (k+1)]
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim *
                                  (i + 1), 2 * i:2 * (i + 1)] = 1
            for j, (merge_pair,
                    highest_ll_k) in enumerate(zip(pairs_to_merge,
                                                   highest_ll)):
                # j + len(mus_ind_not_split) is the new index and k is the old one. We use interleave to create 2 new clusters for each split cluster
                i = j + len(mus_ind_not_merged)
                weights = [
                    class_fc2.weight.data[2 * k:2 * (k + 1), self.hidden_dim *
                                          k:self.hidden_dim * (k + 1)]
                    for k in merge_pair
                ]
                weights = self._initalize_weights_merge(
                    weights,
                    (torch.tensor(highest_ll_k) == merge_pair[1]).item(),
                    init_new_weights)
                bias = [
                    class_fc2.bias.data[2 * k:2 * (k + 1)] for k in merge_pair
                ]
                bias = self._initalize_weights_merge(
                    bias, (torch.tensor(highest_ll_k) == merge_pair[1]).item(),
                    init_new_weights)
                fc2_new_bias.append(bias)
                self.class_fc2.weight.data[2 * i:2 * (i + 1), self.hidden_dim *
                                           i:self.hidden_dim *
                                           (i + 1)] = weights
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim *
                                  (i + 1), 2 * i:2 * (i + 1)] = 1

            self.class_fc2.weight.register_hook(lambda grad: grad.mul_(
                gradient_mask_fc2.T.to(device=self.device)))
            fc2_new_bias = torch.cat(fc2_new_bias)

            # Adjust bias
            fc1_bias_not_merged = class_fc1.bias.data[torch.logical_not(
                merge_decisions).repeat_interleave(self.hidden_dim)]
            fc2_bias_not_merged = class_fc2.bias.data[torch.logical_not(
                merge_decisions).repeat_interleave(2)]

            self.class_fc1.bias.data = torch.cat(
                [fc1_bias_not_merged, fc1_new_bias])
            self.class_fc2.bias.data = torch.cat(
                [fc2_bias_not_merged, fc2_new_bias])
            self.class_fc1.to(device=self.device)
            self.class_fc2.to(device=self.device)

            del class_fc1, class_fc2

    def _initalize_weights_split(self,
                                 weight: torch.Tensor,
                                 init_new_weight: str,
                                 num: int = 2):
        """

        Args:
            weight (torch.Tensor): 
            init_new_weight (str): _description_
            num (int, optional): _description_. Defaults to 2.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if init_new_weight == "same":
            # just duplicate
            dup = weight.reshape(-1, self.hidden_dim,
                                 self.codes_dim).repeat_interleave(num, 0)
            return torch.cat([dup[i] for i in range(dup.size(0))])
        elif init_new_weight == "same_w_noise":
            # just duplicate
            dup = weight.reshape(-1, weight.size(0),
                                 weight.size(1)).repeat_interleave(num, 0)
            return torch.cat([
                dup[i] +
                torch.FloatTensor(dup[i].size(0), dup[i].size(1)).uniform_(
                    -0.01, 0.01).to(device=self.device)
                for i in range(dup.size(0))
            ])
        elif init_new_weight == "random":
            return torch.FloatTensor(weight.shape[0] * num,
                                     weight.shape[1]).uniform_(
                                         -1., 1).to(device=self.device)
        else:
            raise NotImplementedError

    def _initalize_weights_merge(self,
                                 weights_list,
                                 highest_ll_loc,
                                 init_new_weight="highest_ll",
                                 num=2):
        if init_new_weight == "highest_ll":
            # keep the weights of the more likely cluster
            return weights_list[highest_ll_loc]
        elif init_new_weight == "random_choice":
            return weights_list[torch.round(torch.rand(1)).int().item()]
        elif init_new_weight == "random":
            return torch.FloatTensor(weights_list[0].shape[0],
                                     weights_list[0].shape[1]).uniform_(
                                         -1., 1).to(device=self.device)
        else:
            raise NotImplementedError

    def _initalize_bias_split(self, bias_split, init_new_weight):
        if init_new_weight == "same":
            # just duplicate, can think of something more complex later
            return bias_split.repeat(2)
        elif init_new_weight == "same_w_noise":
            # just duplicate, can think of something more complex later
            return bias_split.repeat(2) + torch.FloatTensor(
                bias_split.repeat(2).size(0)).uniform_(
                    -0.01, 0.01).to(device=self.device)
        elif init_new_weight == "random":
            return torch.zeros(bias_split.shape[0] * 2).to(device=self.device)
        else:
            raise NotImplementedError

    def _initalize_bias_merge(self,
                              bias_list,
                              highest_ll,
                              init_new_weight="highest_ll",
                              num=2):
        if init_new_weight == "highest_ll":
            # keep the weights of the more likely cluster
            return bias_list[highest_ll]
        elif init_new_weight == "random":
            return bias_list[torch.round(torch.rand(1)).int().item()]
        else:
            raise NotImplementedError

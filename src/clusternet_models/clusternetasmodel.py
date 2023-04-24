#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, Tensor
from torch.optim import Optimizer
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

from src.clusternet_models.utils.plotting_utils import PlotUtils
from src.clusternet_models.utils.training_utils import training_utils
from src.clusternet_models.utils.clustering_utils.priors import (
    Priors, )
from src.clusternet_models.utils.clustering_utils.clustering_operations import (
    init_mus_and_covs,
    compute_data_covs_hard_assignment,
)
from src.clusternet_models.utils.clustering_utils.split_merge_operations import (
    update_models_parameters_split,
    split_step,
    merge_step,
    update_models_parameters_merge,
)
from src.clusternet_models.models.Classifiers import MLP_Classifier, Subclustering_net

from typing import Dict, Any, Optional, List
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from argparse import Namespace

rank_zero_print = rank_zero_only(print)


class ClusterNetModel(pl.LightningModule):
    def __init__(self,
                 hparams,
                 input_dim: int,
                 init_k: int):
        """The main class of the unsupervised clustering scheme.
        Performs all the training steps.

        Args:
            hparams ([namespace]): model-specific hyperparameters
            input_dim (int): the shape of the input data
            train_dl (DataLoader): The dataloader to train on
            init_k (int): The initial K to start the net with

        """

        super().__init__()
        self.automatic_optimization = False

        self.K: int = init_k
        self.codes_dim: int = input_dim
        # indicator to know whether a split was performed
        self.split_performed: bool = False
        self.merge_performed: bool = False

        self.cluster_loss: str = hparams.cluster_loss
        self.cluster_loss_weight: float = hparams.cluster_loss_weight

        self.start_sub_clustering: int = hparams.start_sub_clustering

        self.how_to_init_mu: str = hparams.how_to_init_mu
        self.how_to_init_mu_sub: str = hparams.how_to_init_mu_sub

        self.use_priors: bool = hparams.use_priors
        self.use_labels_for_eval: bool = hparams.use_labels_for_eval
        self.cov_const = hparams.cov_const
        self.alpha = hparams.alpha
        self.raise_merge_proposals = hparams.raise_merge_proposals
        self.train_cluster_net: int = hparams.train_cluster_net
        self.log_metrics_at_train: bool = hparams.log_metrics_at_train

        self.evaluate_every_n_epochs: int = hparams.evaluate_every_n_epochs

        self.split_init_weights_sub: str = hparams.split_init_weights_sub

        self.cluster_lr: float = hparams.cluster_lr
        self.lr_scheduler: str = hparams.lr_scheduler

        self.init_new_weights: str = hparams.init_new_weights
  


        # initialize cluster_net
        self.cluster_net = MLP_Classifier(hparams,
                                          k=self.K,
                                          codes_dim=self.codes_dim)


        self.training_utils = training_utils(hparams)
        self.prior_sigma_scale: float = hparams.prior_sigma_scale
        self.use_priors = bool(hparams.use_priors)
        self.prior = Priors(
            hparams,
            K=self.K,
            codes_dim=self.codes_dim,
            prior_sigma_scale=self.prior_sigma_scale
        )  # we will use for split and merges even if use_priors is false

        self.save_hyperparameters()
        
        self.training_step_outputs:List[List[Tensor]] = [[],[]]
        self.validation_step_outputs:List[Tensor] = []

    def forward(self, x):
        return self.cluster_net(x)

    def on_train_epoch_start(self):
        rank_zero_print(f"Epoch {self.current_epoch:3d} starts.")
        # get current training_stage
        self.current_training_stage = (
            "gather_codes" if self.current_epoch == 0
            and not hasattr(self, "mus") else "train_cluster_net")
        self.initialize_net_params(stage="train")

        self.split_performed = False
        self.merge_performed = False

    def initialize_net_params(self, stage: str):
        self.codes = []
        if stage == "train":
            if self.current_epoch > 0:
                del self.train_resp, self.train_resp_sub, self.train_gt

            self.train_resp = []
            self.train_resp_sub = []
            self.train_gt = []
        elif stage == "val":
            if self.current_epoch > 0:
                del self.val_resp, self.val_resp_sub, self.val_gt
            self.val_resp = []
            self.val_resp_sub = []
            self.val_gt = []
        else:
            raise NotImplementedError()

    def concatenate_net_params(self, stage: str):
        if len(self.codes) > 0:
            self.codes = torch.cat(self.codes)
        if stage == "train":
            if len(self.train_resp) > 0:
                self.train_resp = torch.cat(
                    self.train_resp)  # (n,dim_features)
            if len(self.train_resp_sub) > 0:
                self.train_resp_sub = torch.cat(self.train_resp_sub)  # (n,2*K)
            if len(self.train_gt) > 0:
                self.train_gt = torch.cat(self.train_gt)  # (n,)
        elif stage == "val":
            if len(self.val_resp) > 0:
                self.val_resp = torch.cat(
                    self.val_resp)  # (n,dim_features)
            if len(self.val_resp_sub) > 0:
                self.val_resp_sub = torch.cat(self.val_resp_sub)  # (n,2*K)
            if len(self.val_gt) > 0:
                self.val_gt = torch.cat(self.val_gt)  # (n,)
        else:
            raise NotImplementedError()

    def training_step(self, batch):
        codes, y = batch
        if self.current_training_stage == "gather_codes":
            return self.only_gather_codes(codes, y)
        elif self.current_training_stage == "train_cluster_net":
            return self.cluster_net_pretraining(codes, y)
        else:
            raise NotImplementedError()

    def only_gather_codes(self, codes: Tensor, y: Tensor):
        """Only log codes for initialization

        Args:
            codes (Tensor): The input data in the latent space
            y (Tensor): The ground truth labels
        """
        self.training_utils.log_codes_and_responses(
                model_codes=self.codes,
                model_gt=self.train_gt,
                model_resp=self.train_resp,
                model_resp_sub=self.train_resp_sub,
                codes=codes,
                y=y,
                logits=None,
            )

    def cluster_net_pretraining(self, codes: Tensor, y: Tensor):
        """Pretraining function for the clustering and subclustering nets.
        At this stage, the only loss is the cluster and subcluster loss.

        Args:
            codes (Tensor): The encoded data samples (n_batch,codes_dim)
            y (Tensor): The ground truth labels (n_batch,)
        """

        clus_opt:Optimizer = self.optimizers()
        clus_shceduler=self.lr_schedulers()

        # if self.trainer.is_last_batch:
        #     rank_zero_print(clus_shceduler.get_last_lr())
        codes = codes.view(-1, self.codes_dim)  # (n_batch,codes_dim)
        logits: Tensor = self.cluster_net(codes)  # (n_batch,K)
        cluster_loss = self.training_utils.cluster_loss_function(
                codes,
                logits,
                model_mus=self.mus,
                K=self.K,
                codes_dim=self.codes_dim,
                model_covs=self.covs
                if self.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                pi=self.pi)

        self.log(
                "cluster_net_train/train/cluster_loss",
                self.cluster_loss_weight * cluster_loss,
                on_step=True,
                on_epoch=False,
            )
        loss = self.cluster_loss_weight * cluster_loss

        clus_opt.zero_grad()
        self.manual_backward(loss)
        clus_opt.step()
        
        if self.trainer.is_last_batch:
            clus_shceduler.step()
        
        if self.start_sub_clustering <= self.current_epoch:
            sublogits = None   
        else:
            sublogits = None


        self.training_utils.log_codes_and_responses(
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
                codes,
                logits.detach(),
                y,
                sublogits=sublogits,
        )

        self.training_step_outputs[0].append(loss.detach().clone().cpu())
        
        return loss

    def on_train_epoch_end(self):
        """Perform logging operations and computes the clusters' and the subclusters' centers.
        Also perform split and merges steps

        Args:
            outputs ([type]): [description]
        """
        # rank_zero_print(f"Epoch {self.current_epoch:3d} ends.")
        self.concatenate_net_params(stage='train')

        if self.current_training_stage == "gather_codes":
            # first time to compute mus
            self.prior.init_priors(self.codes.view(-1, self.codes_dim))

            self.freeze_mus_after_init_until = 0
            self.mus, self.covs, self.pi, init_labels = init_mus_and_covs(
                codes=self.codes.view(-1, self.codes_dim),
                K=self.K,
                how_to_init_mu=self.how_to_init_mu,
                logits=self.train_resp,
                use_priors=self.use_priors,
                prior=self.prior,
                random_state=0,
                device=self.device,
            )
            rank_zero_print(f"Initial pi:{self.pi.tolist()}")
            if self.use_labels_for_eval:
                if (self.train_gt < 0).any():
                    # some samples don't have label, e.g., stl10
                    gt = self.train_gt[self.train_gt > -1]
                    init_labels = init_labels[self.train_gt > -1]
                else:
                    gt = self.train_gt
                if len(gt) > 2 * (10**5):
                    # sample only a portion of the codes
                    gt = gt[:2 * (10**5)]
                init_nmi = normalized_mutual_info_score(gt, init_labels)
                init_ari = adjusted_rand_score(gt, init_labels)
                self.log("cluster_net_train/init_nmi", init_nmi)
                self.log("cluster_net_train/init_ari", init_ari)

        else:
            # add avg loss of all losses
            clus_losses=self.training_step_outputs[0]

            
            avg_clus_loss = torch.stack(clus_losses).mean()
            self.log("cluster_net_train/train/avg_cluster_loss", avg_clus_loss)
            if self.current_epoch >= self.start_sub_clustering:
                pass
                
            self.training_step_outputs[0].clear()


            # Compute mus and perform splits/merges
            perform_split = self.training_utils.should_perform_split(
                self.current_epoch)
            perform_merge = self.training_utils.should_perform_merge(
                self.current_epoch,
                self.split_performed)
            # do not compute the mus in the epoch(s) following a split or a merge

            freeze_mus = self.training_utils.freeze_mus(
                self.current_epoch, self.split_performed
            ) or self.current_epoch <= self.freeze_mus_after_init_until

            if not freeze_mus:
                (
                    self.pi,
                    self.mus,
                    self.covs,
                ) = self.training_utils.comp_cluster_params(
                    self.train_resp,
                    self.codes.view(-1, self.codes_dim),
                    self.K,
                    self.prior,
                )

                rank_zero_print(f"pi:{self.pi.tolist()}")
                
            if (self.start_sub_clustering == self.current_epoch + 1):
                (self.mus_sub,self.covs_sub,self.pi_sub,self.train_resp_sub)=self.training_utils.custom_comp_subcluster_params(self.train_resp,self.codes.view(-1, self.codes_dim),
                    self.K,self.prior)
                rank_zero_print(f"Initial pi_sub:{self.pi_sub}")
            elif (self.start_sub_clustering <= self.current_epoch
                  and not freeze_mus):
                (self.mus_sub,self.covs_sub,self.pi_sub,self.train_resp_sub)=self.training_utils.custom_comp_subcluster_params(self.train_resp,self.codes.view(-1, self.codes_dim),
                    self.K,self.prior)
                rank_zero_print(f"pi_sub:{self.pi_sub}")

            if perform_split and not freeze_mus:
                rank_zero_print("perform splits")
                # perform splits
                self.training_utils.last_performed = "split"
                split_decisions = split_step(
                    self.K, self.codes, self.train_resp, self.train_resp_sub,
                    self.mus, self.mus_sub, self.cov_const,
                    self.alpha, self.prior)
                if split_decisions.any():
                    self.split_performed = True
                    self.perform_split_operations(split_decisions)
            if perform_merge and not freeze_mus:
                rank_zero_print("perform merges")
                # make sure no split and merge step occur in the same epoch
                # perform merges
                # =1 to be one epoch after a split
                self.training_utils.last_performed = "merge"
                mus_to_merge, highest_ll_mus = merge_step(
                    self.mus,
                    self.train_resp,
                    self.codes,
                    self.K,
                    self.raise_merge_proposals,
                    self.cov_const,
                    self.alpha,
                    None,
                    prior=self.prior,
                )
                if len(mus_to_merge) > 0:
                    # there are mus to merge
                    self.merge_performed = True
                    self.perform_merge(mus_to_merge, highest_ll_mus)

            # compute nmi, unique z, etc.
            if self.log_metrics_at_train and self.evaluate_every_n_epochs > 0 and self.current_epoch % self.evaluate_every_n_epochs == 0:
                self.log_clustering_metrics()

        if self.split_performed or self.merge_performed:
            self.update_params_split_merge()
            rank_zero_print("Current number of clusters:", self.K)

        self.log("K", torch.tensor(self.K,dtype=torch.float32))

    def perform_split_operations(self, split_decisions):
        # split_decisions is a list of k boolean indicators of whether we would want to split cluster k
        # update the cluster net to have the new K

        clus_opt:LightningOptimizer = self.optimizers()

        # print([(n,id(p)) for n,p in self.cluster_net.named_parameters()])
        # print([id(k) for k in clus_opt.state.keys()])
        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            # there is still some subtle bugs after resume ...
            try:
                clus_opt.state.pop(p)
            except KeyError:
                pass
        self.cluster_net.update_K_split(split_decisions,
                                        self.init_new_weights,
                                        None)
        clus_opt.param_groups[1]["params"] = list(
            self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        mus_ind_to_split = torch.nonzero(split_decisions.clone().detach(),
                                         as_tuple=False)
        (
            self.mus_new,
            self.covs_new,
            self.pi_new
        ) = update_models_parameters_split(split_decisions,
                                           self.mus,
                                           self.covs,
                                           self.pi,
                                           mus_ind_to_split,
                                           self.mus_sub,
                                           self.covs_sub,
                                           self.pi_sub)
        # update K
        rank_zero_print(
            f"Splitting clusters {np.arange(self.K)[split_decisions.bool().tolist()]}"
        )

        self.K += len(mus_ind_to_split)

        self.mus_ind_to_split = mus_ind_to_split

    def update_subcluster_nets_merge(self, merge_decisions, pairs_to_merge,
                                     highest_ll):
        # update the cluster net to have the new K
        subclus_opt = self.optimizers()

        # remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_merge(
            merge_decisions,
            pairs_to_merge=pairs_to_merge,
            highest_ll=highest_ll,
            init_new_weights=self.merge_init_weights_sub)
        subclus_opt.param_groups[0]["params"] = list(
            self.subclustering_net.parameters())

    def perform_merge(self,
                      mus_lists_to_merge,
                      highest_ll_mus,
                      use_priors=True):
        """A method that performs merges of clusters' centers

        Args:
            mus_lists_to_merge (list): a list of lists, each one contains 2 indices of mus that were chosen to be merged.
            highest_ll_mus ([type]): a list of the highest log likelihood index for each pair of mus
        """

        rank_zero_print(f"Merging clusters {mus_lists_to_merge}")
        mus_lists_to_merge = torch.tensor(mus_lists_to_merge)
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        inds_to_mask[mus_lists_to_merge.flatten()] = 1
        (
            self.mus_new,
            self.covs_new,
            self.pi_new,
            self.mus_sub_new,
            self.covs_sub_new,
            self.pi_sub_new,
        ) = update_models_parameters_merge(
            mus_lists_to_merge,
            inds_to_mask,
            self.K,
            self.mus,
            self.covs,
            self.pi,
            self.mus_sub,
            self.covs_sub,
            self.pi_sub,
            self.codes,
            self.train_resp,
            self.prior,
            use_priors=self.use_priors,
            how_to_init_mu_sub=self.how_to_init_mu_sub,
        )
        # adjust k
        self.K -= len(highest_ll_mus)

        # update the subclustering net
        self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge,
                                          highest_ll_mus)

        # update the cluster net to have the new K
        clus_opt = self.optimizers()

        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)

        # update cluster net
        self.cluster_net.update_K_merge(
            inds_to_mask,
            mus_lists_to_merge,
            highest_ll_mus,
            init_new_weights=self.init_new_weights,
        )
        # add parameters to the optimizer
        clus_opt.param_groups[1]["params"] = list(
            self.cluster_net.class_fc2.parameters())

        self.cluster_net.class_fc2.to(self._device)
        self.mus_inds_to_merge = mus_lists_to_merge

    def configure_optimizers(self):
        # Get all params but last layer
        cluster_params = torch.nn.ParameterList([
            p for n, p in self.cluster_net.named_parameters()
            if "class_fc2" not in n
        ])
        print("====cluster parameters")
        print([n for n, p in self.cluster_net.named_parameters() if "class_fc2" not in n])
        print([n for n, p in self.cluster_net.class_fc2.named_parameters()])
        cluster_net_opt = optim.Adam(cluster_params,
                                     lr=self.cluster_lr)
        # distinct parameter group for the last layer for easy update
        cluster_net_opt.add_param_group(
            {"params": self.cluster_net.class_fc2.parameters()})

        if self.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(
                cluster_net_opt, step_size=20,gamma=0.95)
        elif self.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                cluster_net_opt, mode="min", factor=0.5, patience=4)
        elif self.lr_scheduler == "CosineAnnealingLR":
            cluster_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(cluster_net_opt,T_max=self.trainer.max_epochs,eta_min=self.cluster_lr/10)
        else:
            cluster_scheduler = None

        return ({
                "optimizer": cluster_net_opt,
                "lr_scheduler": {
                "scheduler": cluster_scheduler,
                "monitor": "cluster_net_train/val/cluster_loss"
            }  
                })

    def update_params_split_merge(self):
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.pi = self.pi_new

    def init_covs_and_pis_given_mus(self):
        dis_mat = torch.empty((len(self.codes), self.K))
        for i in range(self.K):
            dis_mat[:, i] = torch.sqrt(
                ((self.codes - self.mus[i])**2).sum(axis=1))
        # get hard assingment
        hard_assign = torch.argmin(dis_mat, dim=1)

        # data params
        vals, counts = torch.unique(hard_assign, return_counts=True)
        if len(counts) < self.K:
            new_counts = []
            for k in range(self.K):
                if k in vals:
                    new_counts.append(counts[vals == k])
                else:
                    new_counts.append(0)
            counts = torch.tensor(new_counts)
        pi = counts / float(len(self.codes))
        data_covs = compute_data_covs_hard_assignment(hard_assign.numpy(),
                                                      self.codes, self.K,
                                                      self.mus.cpu(),
                                                      self.prior)
        if self.use_priors:
            covs = []
            for k in range(self.K):
                codes_k = self.codes[hard_assign == k]
                cov_k = self.prior.compute_post_cov(counts[k],
                                                    codes_k.mean(axis=0),
                                                    data_covs[k])
                covs.append(cov_k)
            covs = torch.stack(covs)
        self.covs = covs
        self.pi = pi

    def log_logits(self, logits):
        for k in range(self.K):
            max_k = logits[logits.argmax(axis=1) == k].detach().cpu().numpy()
            if len(max_k > 0):
                fig = plt.figure(figsize=(10, 3))
                for i in range(len(max_k[:20])):
                    if i == 0:
                        plt.bar(np.arange(self.K),
                                max_k[i],
                                fill=False,
                                label=len(max_k))
                    else:
                        plt.bar(np.arange(self.K), max_k[i], fill=False)
                plt.xlabel("Clusters inds")
                plt.ylabel("Softmax histogram")
                plt.title(f"Epoch {self.current_epoch}: cluster {k}")
                plt.legend()

                # self.logger.log_image(f"cluster_net_train/val/logits_reaction_fig_cluster_{k}", fig)
                plt.close(fig)

    def plot_histograms(self, train=True, for_thesis=False):
        pi = self.pi_new if self.split_performed or self.merge_performed else self.pi

        pi_sub = (self.pi_sub_new if self.split_performed
                  or self.merge_performed else self.pi_sub if
                  self.start_sub_clustering <= self.current_epoch
                  else None)

        fig = PlotUtils.plot_weights_histograms(
            K=self.K,
            pi=pi,
            start_sub_clustering=self.start_sub_clustering,
            current_epoch=self.current_epoch,
            pi_sub=pi_sub,
            for_thesis=for_thesis)
        if for_thesis:
            stage = "val_for_thesis"
        else:
            stage = "train" if train else "val"

        from lightning.pytorch.loggers.logger import DummyLogger
        if not isinstance(self.logger, DummyLogger):
            self.logger.log_image(
                f"cluster_net_train/{stage}/clusters_weights_fig", fig)
        plt.close(fig)

    def log_clustering_metrics(self, stage="train"):
        if stage == "train":
            gt = self.train_gt
            resp = self.train_resp
        elif stage == "val":
            gt = self.val_gt
            resp = self.val_resp
            self.log("cluster_net_train/Networks_k", self.K)
        elif stage == "total":
            gt = torch.cat([self.train_gt, self.val_gt])
            resp = torch.cat([self.train_resp, self.val_resp])

        z = resp.argmax(axis=1).cpu()
        unique_z = len(np.unique(z))
        if len(np.unique(z)) >= 5:
            val, z_top5 = torch.topk(resp, k=5, largest=True)
        else:
            z_top5 = None
        if (gt < 0).any():
            z = z[gt > -1]
            z_top5 = z_top5[gt > -1]
            gt = gt[gt > -1]

        gt_nmi = normalized_mutual_info_score(gt, z)
        ari = adjusted_rand_score(gt, z)
        acc_top5, acc = training_utils.cluster_acc(gt, z, z_top5)

        self.log(f"cluster_net_train/{stage}/{stage}_nmi",
                 gt_nmi,
                 on_epoch=True,
                 on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_ari",
                 ari,
                 on_epoch=True,
                 on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc",
                 acc,
                 on_epoch=True,
                 on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc_top5",
                 acc_top5,
                 on_epoch=True,
                 on_step=False)
        self.log(f"cluster_net_train/{stage}/unique_z",
                 torch.tensor(unique_z,dtype=torch.float32),
                 on_epoch=True,
                 on_step=False)

        if (self.log_metrics_at_train and stage == "train") or \
                (not self.log_metrics_at_train and stage != "train"):
            rank_zero_print(
                f"NMI : {gt_nmi}, ARI: {ari}, ACC: {acc}, ACC-5 {acc_top5}, current K: {unique_z}"
            )

        if self.current_epoch in (0, 1, self.train_cluster_net - 1):
            alt_stage = "start" if self.current_epoch == 1 or self.train_cluster_net % self.current_epoch == 0 else "end"

            if unique_z > 1:
                try:
                    silhouette = silhouette_score(self.codes.cpu(),
                                                  z.cpu().numpy())
                except:
                    silhouette = 0
            else:
                silhouette = 0
            ami = adjusted_mutual_info_score(gt.numpy(), z.numpy())
            (homogeneity, completeness,
             v_measure) = homogeneity_completeness_v_measure(
                 gt.numpy(), z.numpy())

            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_nmi",
                     gt_nmi,
                     on_epoch=True,
                     on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ari",
                     ari,
                     on_epoch=True,
                     on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc",
                     acc,
                     on_epoch=True,
                     on_step=False)
            self.log(
                f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc_top5",
                acc_top5,
                on_epoch=True,
                on_step=False)
            self.log(
                f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_silhouette_score",
                silhouette,
                on_epoch=True,
                on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ami",
                     ami,
                     on_epoch=True,
                     on_step=False)
            self.log(
                f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_homogeneity",
                homogeneity,
                on_epoch=True,
                on_step=False)
            self.log(
                f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_v_measure",
                v_measure,
                on_epoch=True,
                on_step=False)
            self.log(
                f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_completeness",
                completeness,
                on_epoch=True,
                on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_unique_z",
                     unique_z,
                     on_epoch=True,
                     on_step=False)

    def on_save_checkpoint(self, checkpoint) -> None:

        attributes = [
            "train_gt", "train_resp", "train_resp_sub", "mus", "mus_sub",
            "covs", "pi", "freeze_mus_after_init_until", "plot_utils", "prior",
            "pi_sub", "K", 'covs_sub'
        ]

        for attr in attributes:
            if hasattr(self, attr):
                checkpoint[attr] = getattr(self, attr)

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        attributes = [
            "train_gt", "train_resp", "train_resp_sub", "mus", "mus_sub",
            "covs", "pi", "freeze_mus_after_init_until", "plot_utils", "prior",
            "pi_sub", "K", 'covs_sub'
        ]
        maybe_mismateched_parameters = [
            'cluster_net.class_fc1.weight', 'cluster_net.class_fc1.bias',
            'cluster_net.class_fc2.weight', 'cluster_net.class_fc2.bias',
        ]

        for t in maybe_mismateched_parameters:
            state_dict = checkpoint["state_dict"]
            with torch.no_grad():
                self.get_parameter(t).data = torch.empty_like(state_dict[t])
                # print(f"{t} shape:{self.get_parameter(t).shape}")

        for attr in attributes:
            if attr in checkpoint.keys():
                self.__setattr__(attr, checkpoint[attr])

        super().on_load_checkpoint(checkpoint)

       
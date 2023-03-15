#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision.utils import save_image, make_grid

from typing import Optional

from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    compute_pi_k, compute_mus, compute_covs, init_mus_and_covs_sub,
    compute_mus_covs_pis_subclusters)


class training_utils:
    def __init__(self, hparams):
        self.hparams = hparams
        self.pretraining_complete: bool = False
        self.last_performed = "merge"
        self.device = "cuda" if torch.cuda.is_available(
        ) and hparams.gpus is not None else "cpu"

    def should_perform_split(self, current_epoch: int) -> bool:
        # computes whether a split step should be performed in the current epoch
        return (self.hparams.start_splitting <= current_epoch
                and ((current_epoch - self.hparams.start_splitting) %
                     self.hparams.split_merge_every_n_epochs == 0)
                and self.last_performed == "merge")

    def should_perform_merge(self, current_epoch: int, split_performed: bool):
        # computes whether a merge step should be performed in the current epoch
        return (self.hparams.start_merging <= current_epoch
                and ((current_epoch - self.hparams.start_merging) %
                     self.hparams.split_merge_every_n_epochs == 0)
                and not split_performed and self.last_performed == "split")

    def freeze_mus(self, current_epoch: int, split_performed: bool) -> bool:
        if (current_epoch < self.hparams.start_computing_params or
            (self.hparams.compute_params_every != 1
             and current_epoch % self.hparams.compute_params_every != 0)):
            return True
        else:
            split_occured = torch.tensor([
                self.should_perform_split(current_epoch - epoch)
                for epoch in range(
                    1,
                    self.hparams.freeze_mus_submus_after_splitmerge + 1,
                    1,
                )
            ]).any()
            merge_occured = torch.tensor([
                self.should_perform_merge(current_epoch - epoch,
                                          split_performed)
                for epoch in range(
                    1,
                    self.hparams.freeze_mus_submus_after_splitmerge + 1,
                    1,
                )
            ]).any()
            return split_occured or merge_occured

    def comp_cluster_params(self, train_resp, codes, pi, K:int, prior=None):
        # compute pi
        pi = compute_pi_k(train_resp,
                          prior=prior if self.hparams.use_priors else None)
        mus = compute_mus(
            codes=codes,
            logits=train_resp,
            pi=pi,
            K=K,
            how_to_compute_mu=self.hparams.how_to_compute_mu,
            use_priors=self.hparams.use_priors,
            prior=prior,
        )

        covs = compute_covs(
            logits=train_resp,
            codes=codes,
            K=K,
            mus=mus,
            use_priors=self.hparams.use_priors,
            prior=prior,
        )
        return pi, mus, covs

    def comp_subcluster_params(
        self,
        train_resp,
        train_resp_sub,
        codes,
        K,
        n_sub,
        mus_sub,
        covs_sub,
        pi_sub,
        prior=None,
    ):

        mus_sub, covs_sub, pi_sub = compute_mus_covs_pis_subclusters(
            codes=codes,
            logits=train_resp,
            logits_sub=train_resp_sub,
            mus_sub=mus_sub,
            K=K,
            n_sub=n_sub,
            use_priors=self.hparams.use_priors,
            prior=prior)
        return pi_sub, mus_sub, covs_sub

    def init_subcluster_params(self,
                               train_resp,
                               train_resp_sub,
                               codes,
                               K,
                               n_sub,
                               prior=None):
        mus_sub, covs_sub, pi_sub = [], [], []
        for k in range(K):
            mus, covs, pis = init_mus_and_covs_sub(
                codes=codes,
                k=k,
                n_sub=n_sub,
                how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
                logits=train_resp,
                logits_sub=train_resp_sub,
                prior=prior,
                use_priors=self.hparams.use_priors,
                device=self.device)
            mus_sub.append(mus)
            covs_sub.append(covs)
            pi_sub.append(pis)
        mus_sub = torch.cat(mus_sub)
        covs_sub = torch.cat(covs_sub)
        pi_sub = torch.cat(pi_sub)

        return pi_sub, mus_sub, covs_sub

    def cluster_loss_function(
        self,
        c,
        r,
        model_mus,
        K,
        codes_dim,
        model_covs=None,
        pi=None,
    ):
        if self.hparams.cluster_loss == "isotropic":
            # Isotropic
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (r_tag * ((torch.norm(
                C_tag - mus_tag.to(device=self.device), dim=1))**2)).mean()

        elif self.hparams.cluster_loss == "diag_NIG":
            # NIG prior
            # K * N, D
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            sigmas = torch.sqrt(model_covs).repeat(c.shape[0], 1)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (r_tag * ((torch.norm(
                (C_tag - mus_tag.to(device=self.device)) /
                sigmas.to(device=self.device),
                dim=1))**2)).mean()

        elif self.hparams.cluster_loss == "KL_GMM_2":
            r_gmm = []
            for k in range(K):
                gmm_k = MultivariateNormal(
                    model_mus[k].double().to(device=self.device),
                    model_covs[k].double().to(device=self.device))
                prob_k = gmm_k.log_prob(c.detach().double())
                r_gmm.append((prob_k + torch.log(pi[k])).double())
            r_gmm = torch.stack(r_gmm).T
            max_values, _ = r_gmm.max(axis=1, keepdim=True)
            r_gmm -= torch.log(
                torch.exp((r_gmm - max_values)).sum(axis=1,
                                                    keepdim=True)) + max_values
            r_gmm = torch.exp(r_gmm)
            eps = 0.00001
            r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
            r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)

            return nn.KLDivLoss(reduction="batchmean")(
                torch.log(r),
                r_gmm.float().to(device=self.device),
            )

        raise NotImplementedError("No such loss")

    def subcluster_loss_function_new(self,
                                     codes,
                                     logits,
                                     subresp,
                                     K,
                                     n_sub,
                                     mus_sub,
                                     covs_sub=None,
                                     pis_sub=None):
        if self.hparams.subcluster_loss == "isotropic":
            # Isotropic

            C_tag = codes.repeat(1, 2 * K).view(-1, codes.size(1))
            mus_tag = mus_sub.repeat(codes.shape[0], 1)
            r_tag = subresp.flatten()
            return (r_tag *
                    ((torch.norm(C_tag - mus_tag.to(device=self.device),
                                 dim=1))**2)).sum() / float(len(codes))

        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k:2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        gmm_k = MultivariateNormal(
                            mus_sub[2 * k +
                                    k_sub].double().to(device=self.device),
                            covs_sub[2 * k +
                                     k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(codes_k.detach().double())
                        r_gmm.append(
                            (prob_k +
                             torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(
                        torch.exp((r_gmm - max_values)).sum(
                            axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1,
                                                              keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r),
                        r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(
                            codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(covs_sub[2 * k + k_sub].repeat(
                            codes_k.shape[0], 1))
                        loss += (r * ((torch.norm(
                            (codes_k - mus_tag.to(device=self.device) /
                             sigma_sub.to(device=self.device)),
                            dim=1,
                        ))**2)).sum()
            return loss

        raise NotImplementedError("No such loss!")

    @staticmethod
    def _best_cluster_fit(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_assignment(w.max() - w)
        map_dict = {}
        for j in range(len(col_ind)):
            map_dict[col_ind[j]] = row_ind[j]
        y_true_new = np.array([map_dict[i] for i in y_true])
        return y_true_new, row_ind, col_ind, w

    @staticmethod
    def cluster_acc(y_true, y_pred, y_pred_top5=None):
        y_true_new, row_ind, col_ind, w = training_utils._best_cluster_fit(
            y_true.numpy(), y_pred.numpy())
        if y_pred_top5 is not None:
            y_true_new = torch.from_numpy(y_true_new).unsqueeze(0).repeat(5, 1)
            acc_top5 = (y_pred_top5.T == y_true_new).any(
                axis=0).sum() * 1.0 / y_pred.numpy().size
            acc_top5 = acc_top5.item()
        else:
            acc_top5 = 0.

        return acc_top5, np.round(
            w[row_ind, col_ind].sum() * 1.0 / y_pred.numpy().size, 5)

    @staticmethod
    def log_codes_and_responses(
        model_codes,
        model_gt,
        model_resp,
        model_resp_sub,
        codes,
        logits: Tensor,  # output of the model (n,dim_features)
        y: Tensor,  # ground truth labels
        sublogits: Optional[Tensor] = None,
        stage="train",
    ):
        """A function to log data used to compute model's parameters.

        Args:
            codes (torch.tensor): the current batch codes (in emedding space)
            logits (torch.tensor): the clustering net responses to the codes
            y (torch.tensor): the ground truth labels
            sublogits ([type], optional): [description]. Defaults to None. The subclustering nets response to the codes
        """
        if model_gt == []:
            # first batch of the epoch
            if codes is not None:
                model_codes = codes.detach().cpu()
            model_gt = y.detach().cpu()
            if logits is not None:
                model_resp = logits.detach().cpu()
            if sublogits is not None:
                model_resp_sub = sublogits.detach().cpu()
        else:
            if codes is not None:
                model_codes = torch.cat([model_codes, codes.detach().cpu()])
            model_gt = torch.cat([model_gt, y.detach().cpu()])
            if logits is not None:
                model_resp = torch.cat([model_resp, logits.detach().cpu()])
            if sublogits is not None:
                model_resp_sub = torch.cat(
                    [model_resp_sub, sublogits.detach().cpu()])
        return model_codes, model_gt, model_resp, model_resp_sub

    
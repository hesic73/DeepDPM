#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import numpy as np
from math import lgamma

from typing import Optional

# from kmeans_pytorch import kmeans as GPU_KMeans
from src.clusternet_models.utils.miscellaneous import GPU_KMeans

from sklearn.neighbors import NearestNeighbors
from src.clusternet_models.utils.clustering_utils.clustering_operations import (
    _create_subclusters, compute_data_covs_soft_assignment,
    init_mus_and_covs_sub, comp_subclusters_params_min_dist)

from lightning.pytorch.utilities.rank_zero import rank_zero_only

rank_zero_print = rank_zero_only(print)


def log_Hastings_ratio_split(alpha: float, N_k_1: int, N_k_2: int,
                             log_ll_k_1: float, log_ll_k_2: float,
                             log_ll_k: float) -> float:
    N_k = N_k_1 + N_k_2
    # each subcluster is not empty
    log_H = (np.log(alpha) + lgamma(N_k_1) + log_ll_k_1 + lgamma(N_k_2) +
             log_ll_k_2) - (lgamma(N_k) + log_ll_k)
    rank_zero_print(
        f"log_H={log_H},N_k_1={N_k_1},N_k_2={N_k_2},log_ll_k_1={log_ll_k_1},log_ll_k_2={log_ll_k_2},log_ll_k={log_ll_k}"
    )
    return log_H


def log_Hastings_ratio_merge(alpha: float, N_k_1: int, N_k_2: int,
                             log_ll_k_1: float, log_ll_k_2: float,
                             log_ll_k: float) -> float:
    # use log for overflows
    if N_k_1 == 0:
        lgamma_1 = 0
    else:
        lgamma_1 = lgamma(N_k_1)
    if N_k_2 == 0:
        lgamma_2 = 0
    else:
        lgamma_2 = lgamma(N_k_2)
    # Hastings ratio in log space
    N_k = N_k_1 + N_k_2
    if N_k > 0:
        log_H = ((lgamma(N_k) - (np.log(alpha) + lgamma_1 + lgamma_2)) +
                 (log_ll_k - (log_ll_k_1 + log_ll_k_2)))
    else:
        log_H = torch.zeros(1)

    return log_H


def split_rule(k: int,
               codes,
               logits,
               logits_sub,
               mus,
               mus_sub,
               cov_const,
               alpha,
               prior=None):
    # look at the points assigned to k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    if len(codes_k) < 5:
        # empty cluster
        return -1e8

    
    # subclusters hard assignment
    sub_assignment = logits_sub[codes_ind, :].argmax(-1)
    codes_k_1 = codes_k[sub_assignment == 2 * k]
    codes_k_2 = codes_k[sub_assignment == 2 * k + 1]

    if len(codes_k_1) <= 5 or len(codes_k_2) <= 5:
        # small subclusters
        return -1e8

    # compute log marginal likelihood
    log_ll_k = prior.log_marginal_likelihood(codes_k, mus[k])
    log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus_sub[2 * k])
    log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus_sub[2 * k + 1])

    N_k_1 = len(codes_k_1)
    N_k_2 = len(codes_k_2)

    if N_k_1 == 0 or N_k_2 == 0:
        return -1e8

    # use log for overflows
    # Hastings ratio in log space
    return log_Hastings_ratio_split(alpha, N_k_1, N_k_2, log_ll_k_1,
                                    log_ll_k_2, log_ll_k)


def compute_split_log_ll(mu, mus_sub_1, mus_sub_2, cov_const, codes_k,
                         codes_k_1, codes_k_2):
    D = len(mu)
    dist_k = torch.distributions.multivariate_normal.MultivariateNormal(
        mu,
        torch.eye(D) * cov_const)
    dist_k_1 = torch.distributions.multivariate_normal.MultivariateNormal(
        mus_sub_1,
        torch.eye(D) * cov_const)
    dist_k_2 = torch.distributions.multivariate_normal.MultivariateNormal(
        mus_sub_2,
        torch.eye(D) * cov_const)

    log_ll_k = dist_k.log_prob(codes_k).sum()
    log_ll_k_1 = (dist_k_1.log_prob(codes_k_1)).sum()
    log_ll_k_2 = (dist_k_2.log_prob(codes_k_2)).sum()

    return log_ll_k, log_ll_k_1, log_ll_k_2


# extern
def split_step(K: int,
               codes,
               logits,
               logits_sub,
               mus,
               mus_sub,
               cov_const,
               alpha,
               split_prob: Optional[float],
               prior=None):

    if split_prob is not None:
        split_decisions = torch.full((K, ), split_prob)
        return split_decisions > torch.rand_like(split_decisions)

    split_decisions = torch.empty(K, dtype=torch.float32)
    for k in range(K):
        split_decisions[k] = split_rule(k,
                                        codes,
                                        logits,
                                        logits_sub,
                                        mus,
                                        mus_sub,
                                        cov_const,
                                        alpha,
                                        prior=prior)
    rank_zero_print(f"split decisions:{split_decisions.tolist()}")
    split_decisions = torch.exp(split_decisions) > torch.rand_like(
        split_decisions)
    return split_decisions


def update_clusters_params_split(mus, covs, pi, mus_ind_to_split,
                                 split_decisions, mus_sub, covs_sub, pi_sub):
    """This function is used to compute the new model parameters following a split

    Args:
        mus ([torch.tensor]): The mus before the split
        covs ([torch.tensor]): The covs before the split
        pi ([torch.tensor]): The pis before the split
        mus_ind_to_split ([list]): A list of the mus that were chosen to be split
        split_decisions ([list]): A boolean list of len(mus) with True where mus_ind was split
        mus_sub ([type]): The subclusters' mus before the split

    Returns:
        mus_new ([torch.tensor]), covs_new ([torch.tensor]), pi_new ([torch.tensor]): The new parameters
    """

    mus_new = mus[torch.logical_not(split_decisions)]
    covs_new = covs[torch.logical_not(split_decisions)]
    pi_new = pi[torch.logical_not(split_decisions)]

    mus_to_add, covs_to_add, pis_to_add = [], [], []
    for k in mus_ind_to_split:
        mus_to_add.extend([mus_sub[2 * k], mus_sub[2 * k + 1]])
        covs_to_add.extend([covs_sub[2 * k], covs_sub[2 * k + 1]])
        pis_to_add.extend([pi_sub[2 * k], pi_sub[2 * k + 1]])

    mus_new = torch.cat([mus_new, torch.cat(mus_to_add)])
    covs_new = torch.cat([covs_new, torch.cat(covs_to_add)])
    pi_new = torch.cat([pi_new, torch.cat(pis_to_add)])

    return mus_new, covs_new, pi_new


def update_subclusters_params_split(mus_sub,
                                    covs_sub,
                                    pi_sub,
                                    mus_ind_to_split,
                                    split_decisions,
                                    codes,
                                    logits,
                                    logits_sub,
                                    how_to_init_mu_sub,
                                    prior,
                                    use_priors=True):
    mus_sub_new = mus_sub[torch.logical_not(split_decisions).repeat_interleave(
        2)]
    covs_sub_new = covs_sub[torch.logical_not(
        split_decisions).repeat_interleave(2)]
    pi_sub_new = pi_sub[torch.logical_not(split_decisions).repeat_interleave(
        2)]
    mus_sub_to_add, covs_sub_to_add, pis_sub_to_add = [], [], []
    for k in mus_ind_to_split:
        (
            new_mus_sub_1,
            new_covs_sub_1,
            new_pis_1,
        ) = _create_subclusters(k_sub=2 * k,
                                codes=codes,
                                logits=logits,
                                logits_sub=logits_sub,
                                mus_sub=mus_sub,
                                pi_sub=pi_sub,
                                how_to_init_mu_sub=how_to_init_mu_sub,
                                prior=prior,
                                use_priors=use_priors)
        new_mus_sub_2, new_covs_sub_2, new_pis_2 = _create_subclusters(
            k_sub=2 * k + 1,
            codes=codes,
            logits=logits,
            logits_sub=logits_sub,
            mus_sub=mus_sub,
            pi_sub=pi_sub,
            how_to_init_mu_sub=how_to_init_mu_sub,
            prior=prior,
            use_priors=use_priors)
        mus_sub_to_add.extend([new_mus_sub_1, new_mus_sub_2])
        covs_sub_to_add.extend([new_covs_sub_1, new_covs_sub_2])
        pis_sub_to_add.extend([new_pis_1, new_pis_2])

    mus_sub_new = torch.cat([mus_sub_new, torch.cat(mus_sub_to_add)])
    covs_sub_new = torch.cat([covs_sub_new, torch.cat(covs_sub_to_add)])
    pi_sub_new = torch.cat([pi_sub_new, torch.cat(pis_sub_to_add)])

    return mus_sub_new, covs_sub_new, pi_sub_new


def update_models_parameters_split(split_decisions, mus, covs, pi,
                                   mus_ind_to_split, mus_sub, covs_sub, pi_sub,
                                   codes, logits, logits_sub, 
                                   how_to_init_mu_sub, prior, use_priors):
    mus_ind_to_split = torch.nonzero(split_decisions, as_tuple=False)
    # update the mus, covs and pis
    mus_new, covs_new, pi_new = update_clusters_params_split(
        mus, covs, pi, mus_ind_to_split, split_decisions, mus_sub, covs_sub,
        pi_sub)
    # update the submus, subcovs and subpis
    mus_sub_new, covs_sub_new, pi_sub_new = update_subclusters_params_split(
        mus_sub,
        covs_sub,
        pi_sub,
        mus_ind_to_split,
        split_decisions,
        codes,
        logits,
        logits_sub,
        how_to_init_mu_sub,
        prior,
        use_priors=use_priors)
    return mus_new, covs_new, pi_new, mus_sub_new, covs_sub_new, pi_sub_new


def update_clusters_params_merge(
    mus_lists_to_merge,
    inds_to_mask,
    mus,
    covs,
    pi,
    K,
    codes,
    logits,
    prior,
    use_priors,
    how_to_init_mu_sub,
):
    mus_not_merged = mus[torch.logical_not(inds_to_mask)]
    covs_not_merged = covs[torch.logical_not(inds_to_mask)]
    pis_not_merged = pi[torch.logical_not(inds_to_mask)]
    # compute new clusters' centers:
    mus_merged, covs_merged, pi_merged = [], [], []
    for pair in mus_lists_to_merge:
        N_k_1 = (logits.argmax(-1) == pair[0]).sum().type(torch.float32)
        N_k_2 = (logits.argmax(-1) == pair[1]).sum().type(torch.float32)
        N_k = N_k_1 + N_k_2

        if N_k > 0:
            mus_mean = (N_k_1 / N_k) * mus[pair[0]] + (N_k_2 /
                                                       N_k) * mus[pair[1]]
            cov_new = compute_data_covs_soft_assignment(
                logits=(logits[:, pair[0]] + logits[:, pair[1]]).reshape(
                    -1, 1),
                codes=codes,
                K=1,
                mus=mus_mean,
                prior_name=prior.name)
        else:
            # in case both are empty clusters
            mus_mean = mus[pair].mean(axis=0)
            cov_new = covs[pair[0]].unsqueeze(0)

        pi_new = (pi[pair[0]] + pi[pair[1]]).reshape(1)

        if use_priors:
            r_k = (logits[:, pair[0]] + logits[:, pair[1]]).sum(axis=0)
            cov_new = prior.compute_post_cov(r_k, mus_mean, cov_new)
            mus_mean = prior.compute_post_mus(pi_new * len(codes), mus_mean)

        mus_merged.append(mus_mean)
        covs_merged.append(cov_new)
        pi_merged.append(pi_new)

    mus_merged = torch.stack(mus_merged).squeeze(1)
    covs_merged = torch.stack(covs_merged).squeeze(1)
    pi_merged = torch.stack(pi_merged).squeeze(1)

    mus_new = torch.cat([mus_not_merged, mus_merged])
    covs_new = torch.cat([covs_not_merged, covs_merged])
    pi_new = torch.cat([pis_not_merged, pi_merged])

    return mus_new, covs_new, pi_new


def update_subclusters_params_merge(mus_lists_to_merge,
                                    inds_to_mask,
                                    mus,
                                    covs,
                                    pi,
                                    mus_sub,
                                    covs_sub,
                                    pi_sub,
                                    codes,
                                    logits,
                                    how_to_init_mu_sub,
                                    prior,
                                    use_priors=True):
    # update sub_mus
    mus_sub_not_merged = mus_sub[torch.logical_not(
        inds_to_mask.repeat_interleave(2))]
    covs_sub_not_merged = covs_sub[torch.logical_not(
        inds_to_mask.repeat_interleave(2))]
    pi_sub_not_merged = pi_sub[torch.logical_not(
        inds_to_mask.repeat_interleave(2))]

    mus_sub_merged, covs_sub_merged, pi_sub_merged = [], [], []
    for n_merged in range(len(mus_lists_to_merge)):
        codes_merged = codes[torch.logical_or(
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][0]),
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][1]))]
        if len(codes_merged) <= 5:
            # Both clusters are empty or have very few points
            mus_sub_merged.append(mus[mus_lists_to_merge[n_merged].flatten()])
            covs_sub_merged.append(
                covs[mus_lists_to_merge[n_merged].flatten()])
            pi_sub_merged.append(pi[mus_lists_to_merge[n_merged].flatten()])
        else:
            mus_sub_k, covs_sub_k, pi_sub_k = init_mus_and_covs_sub(
                codes_merged,
                k=0,
                how_to_init_mu_sub=how_to_init_mu_sub,
                logits=torch.zeros(len(codes_merged), 1),
                prior=prior,
                use_priors=use_priors)
            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)
    mus_sub_merged = torch.cat(mus_sub_merged)
    covs_sub_merged = torch.cat(covs_sub_merged)
    pi_sub_merged = torch.cat(pi_sub_merged)

    mus_sub_new = torch.cat([mus_sub_not_merged, mus_sub_merged])
    covs_sub_new = torch.cat([covs_sub_not_merged, covs_sub_merged])
    pi_sub_new = torch.cat([pi_sub_not_merged, pi_sub_merged])

    return mus_sub_new, covs_sub_new, pi_sub_new


def update_models_parameters_merge(
    mus_lists_to_merge,
    inds_to_mask,
    K,
    mus,
    covs,
    pi,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    prior,
    use_priors,
    how_to_init_mu_sub,
):

    mus_new, covs_new, pi_new = update_clusters_params_merge(
        mus_lists_to_merge,
        inds_to_mask,
        mus,
        covs,
        pi,
        K,
        codes,
        logits,
        prior,
        use_priors,
        how_to_init_mu_sub,
    )
    mus_sub_new, covs_sub_new, pi_sub_new = update_subclusters_params_merge(
        mus_lists_to_merge,
        inds_to_mask,
        mus,
        covs,
        pi,
        mus_sub,
        covs_sub,
        pi_sub,
        codes,
        logits,
        how_to_init_mu_sub,
        prior,
        use_priors=use_priors)
    return mus_new, covs_new, pi_new, mus_sub_new, covs_sub_new, pi_sub_new


# extern
def merge_step(mus,
               logits,
               codes,
               K,
               raise_merge_proposals,
               cov_const,
               alpha,
               merge_prob,
               h_merge="pairs",
               prior=None):
    """
    we will cluster all the mus into @h_merge clusters.
    A possible h_param for h_merge should be a function of the current K, e.g., sqrt(K) or something like that
    Then we will perform merges within each cluster of mus (if two mus where not assigned to the same cluster,
    they will not be considered for merging)
    For all the clusters (mus) that are in the same cluster, we will take a random permutation
    and consider merges by pairs (0&1, 2&3, ...)
    """

    if h_merge == "pairs":
        n_cluster = K / 2
    # mus to merge is a list of lists of 2 mus indices, meaning each list contains pairs of mus to merge
    # highest ll mus contains for each pair, the index of the one with the highest likelihood
    mus_to_merge, highest_ll_mus = [], []

    if raise_merge_proposals == "kmeans":
        labels, cluster_centers = GPU_KMeans(X=mus.detach(),
                                             num_clusters=n_cluster,
                                             device=torch.device('cuda:0'))

        for i in range(n_cluster):
            chosen_ind = torch.nonzero(labels == i, as_tuple=False)
            perm = torch.randperm(len(chosen_ind))
            # shuffle mus before choosing merges
            merge_decision, highest_ll = merge_rule(mus,
                                                    logits,
                                                    codes,
                                                    chosen_ind[perm],
                                                    alpha,
                                                    cov_const,
                                                    merge_prob,
                                                    prior=prior)

            merge_decision = [
                torch.exp(l.float()) > torch.rand(1) for l in merge_decision
            ]
            # merge decision returns a boolean array with the decision on whether to merge each pair
            # so, if we had N chosen mus, merge decision will be of size N/2. If it's true at 0
            # then we will merge the chosen mus at [0, 1]
            for n_pair in range(len(merge_decision)):
                if merge_decision[n_pair]:
                    mus_to_merge.append([
                        chosen_ind[perm][2 * n_pair:2 * n_pair + 2][0][0],
                        chosen_ind[perm][2 * n_pair:2 * n_pair + 2][1][0],
                    ])
                    highest_ll_mus.append(highest_ll[n_pair])

    elif (raise_merge_proposals == "brute_force_NN"
          or raise_merge_proposals == "brute_force_NN_with_bad"):
        n_neighbors = min(3, K)
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(mus)
        A = torch.tensor(neigh.kneighbors_graph(mus).toarray()) - torch.eye(
            len(mus))
        neigh_inds_per_cluster = torch.nonzero(A, as_tuple=False)
        keys = np.arange(len(mus))
        mus_to_consider_to_merge = dict(zip(keys, keys))
        for proposed_pair in neigh_inds_per_cluster:
            p_0 = proposed_pair[0].item()
            p_1 = proposed_pair[1].item()
            if p_0 in mus_to_consider_to_merge.keys(
            ) and p_1 in mus_to_consider_to_merge.keys():
                # did not merge before
                merge_decision, highest_ll = merge_rule(mus,
                                                        logits,
                                                        codes,
                                                        proposed_pair,
                                                        alpha,
                                                        cov_const,
                                                        merge_prob,
                                                        prior=prior)
                merge_decision = [
                    torch.exp(l.float()) > torch.rand(1)
                    for l in merge_decision
                ]
                if merge_decision[0]:
                    # merge is accepted
                    mus_to_consider_to_merge.pop(p_0)
                    mus_to_consider_to_merge.pop(p_1)
                    mus_to_merge.append([p_0, p_1])
                    highest_ll_mus.append(highest_ll)

        if raise_merge_proposals == "brute_force_NN_with_bad":
            # add bad mus for sanity check
            for i in range(len(mus)):
                neighbors_ind = A[i, :]
                # sample a neighbors that is not close
                sampled = torch.randint(len(mus), size=(1, )).item()
                flag = neighbors_ind[sampled]
                while flag:
                    sampled = torch.randint(len(mus), size=(1, )).item()
                    flag = neighbors_ind[sampled]
                merge_decision, highest_ll = merge_rule(mus,
                                                        logits,
                                                        codes,
                                                        torch.tensor(
                                                            [i, sampled]),
                                                        alpha,
                                                        cov_const,
                                                        merge_prob,
                                                        prior=prior)
                merge_decision = [
                    torch.exp(l) > torch.rand(1) for l in merge_decision
                ]
    return mus_to_merge, highest_ll_mus


def merge_rule(mus,
               logits,
               codes,
               k_inds,
               alpha,
               cov_const,
               merge_prob,
               prior=None):
    """
    Gets an input a random permutation of indices of the clusters to consider merge.
    We will consider merges of pairs.
    Returns:
    (1) boolean array of size len(k_inds)//2 with the merge decision for every pair
    (2) a list of the indices of the clusterwith the highest likelihood from each pair
    """
    decisions = []
    highest_ll = []

    for i in range(0, len(k_inds), 2):
        # for each pair do
        k_1 = k_inds[i]
        if len(k_inds) - 1 == i:
            # only one cluster
            decisions.append(False)
            highest_ll.append(k_inds[i])
            return decisions, highest_ll
        k_2 = k_inds[i + 1]

        codes_ind_k1 = logits.argmax(-1) == k_1
        codes_ind_k2 = logits.argmax(-1) == k_2
        codes_ind_k = torch.logical_or(codes_ind_k1, codes_ind_k2)

        codes_k_1 = codes[codes_ind_k1]
        codes_k_2 = codes[codes_ind_k2]
        codes_k = codes[codes_ind_k]

        N_k_1 = len(codes_k_1)
        N_k_2 = len(codes_k_2)
        N_k = N_k_1 + N_k_2

        if N_k > 0:
            mus_mean = (N_k_1 / N_k) * mus[k_1] + (N_k_2 / N_k) * mus[k_2]
        else:
            # in case both are empty clusters
            mus_mean = torch.mean(torch.stack([mus[k_1], mus[k_2]]), axis=0)
        if prior is None:
            (
                log_ll_k,
                log_ll_k_1,
                log_ll_k_2,
            ) = compute_split_log_ll(mus_mean, mus[k_1], mus[k_2], cov_const,
                                     codes_k, codes_k_1, codes_k_2)
        else:
            log_ll_k = prior.log_marginal_likelihood(codes_k, mus_mean)
            log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus[k_1])
            log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus[k_2])
        prob = merge_prob or log_Hastings_ratio_merge(
            alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k)
        decisions.append(prob)
        highest_ll.append(k_inds[i:i + 2][int(log_ll_k_1 < log_ll_k_2)])

    rank_zero_print(f"merge decisions:{decisions}")
    return decisions, highest_ll

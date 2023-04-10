#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from typing import Optional
import torch
from torch import mvlgamma
from torch import lgamma
from torch import Tensor
import numpy as np


class Priors:
    '''
    A prior that will hold the priors for all the parameters.
    '''

    def __init__(self,
                 hparams,
                 K,
                 codes_dim,
                 counts=10,
                 prior_sigma_scale=None):
        self.name = "prior_class"
        self.pi_prior_type = hparams.pi_prior
        if hparams.pi_prior:
            self.pi_prior = Dirichlet_prior(K, hparams.pi_prior, counts)
        else:
            self.pi_prior = None
        if hparams.prior == "NIW":
            self.mus_covs_prior = NIW_prior(hparams, prior_sigma_scale)
        elif hparams.prior == "NIG":
            self.mus_covs_prior = NIG_prior(hparams, codes_dim)
        self.name = self.mus_covs_prior.name
        self.pi_counts = hparams.prior_dir_counts

    def update_pi_prior(self, K_new, counts=10, pi_prior=None):
        # pi_prior = None- keep the same pi_prior type
        if self.pi_prior:
            if pi_prior:
                self.pi_prioir = Dirichlet_prior(K_new, pi_prior, counts)
            self.pi_prior = Dirichlet_prior(K_new, self.pi_prior_type, counts)

    def comp_post_counts(self, counts):
        if self.pi_prior:
            return self.pi_prior.comp_post_counts(counts)
        else:
            return counts

    def comp_post_pi(self, pi:Tensor)->Tensor:
        if self.pi_prior:
            return self.pi_prior.comp_post_pi(pi, self.pi_counts)
        else:
            return pi

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()

    def init_priors(self, codes:Tensor):
        return self.mus_covs_prior.init_priors(codes)

    def compute_params_post(self, codes_k, mu_k):
        return self.mus_covs_prior.compute_params_post(codes_k, mu_k)

    def compute_post_mus(self, N_ks:Tensor, data_mus:Tensor)->Tensor:
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k)

    def log_marginal_likelihood(self, codes_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(codes_k, mu_k)


class Dirichlet_prior:
    def __init__(self, K, pi_prior="uniform", counts=10):
        self.name = "Dirichlet_dist"
        self.K: int = K
        self.counts = counts
        if pi_prior == "uniform":
            self.p_counts = torch.ones(K) * counts
            self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts=None):
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi:Tensor, counts:Optional[float]=None)->Tensor:
        if counts is None:
            # counts = 0.001
            counts = 0.1
            # counts = 0.0
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


@torch.jit.script
def NIW_log_marginal_likelihood(D: int, N_k: int, nu: float, nu_star: float, psi: Tensor, psi_star: Tensor, kappa: float, kappa_star: float) -> Tensor:
    return (
        -(N_k * D / 2.0) * torch.log(torch.tensor(np.pi)) +
        mvlgamma(torch.tensor(nu_star / 2.0), D) -
        mvlgamma(torch.tensor(nu / 2.0), D) +
        (nu / 2.0) * torch.logdet(psi) -
        (nu_star / 2.0) * torch.logdet(psi_star) + (D / 2.0) *
        torch.log(torch.tensor(kappa/kappa_star)) +
        (D/16 / 2.0) * (nu * torch.log(torch.tensor(nu)) -
                     nu_star * torch.log(torch.tensor(nu_star)))
    )


@torch.jit.script
def NIW_compute_params_post(kappa: float, nu: float, mu_0: Tensor, psi: Tensor, codes_k: Tensor, mu_k: Tensor):
    N_k = len(codes_k)
    sum_k = codes_k.sum(dim=0)
    kappa_star = kappa + N_k
    nu_star = nu + N_k
    mu_0_star = (mu_0 * kappa + sum_k) / kappa_star
    codes_minus_mu = codes_k - mu_k
    S = codes_minus_mu.T @ codes_minus_mu
    psi_star = (
        psi + S + (kappa * N_k / kappa_star) *
        (mu_k - mu_0).unsqueeze(1) @ (mu_k - mu_0).unsqueeze(0)
    )
    return kappa_star, nu_star, mu_0_star, psi_star


@torch.jit.script
def NIW_compute_post_mus(mu_0: Tensor, kappa: float, N_ks: Tensor, data_mus: Tensor):
    return ((N_ks * data_mus) + (kappa * mu_0)) / (N_ks + kappa)


@torch.jit.script
def NIW_compute_post_cov(kappa: float, nu: float, psi: Tensor, mu_0: Tensor, N_k: int, mu_k: Tensor, data_cov_k: Tensor):
    D = len(mu_k)
    return (psi + data_cov_k * N_k  # unnormalize
            + (((kappa * N_k) / (kappa + N_k)) *
               ((mu_k - mu_0).unsqueeze(1) *
                (mu_k - mu_0).unsqueeze(0)))) / (nu + N_k + D + 2)


class NIW_prior:
    """A class used to store niw parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    """

    def __init__(self, hparams, prior_sigma_scale=None):
        self.name = "NIW"
        self.prior_mu_0_choice: str = hparams.prior_mu_0
        self.prior_sigma_choice: str = hparams.prior_sigma_choice
        self.prior_sigma_scale: float = prior_sigma_scale or hparams.prior_sigma_scale
        self.niw_kappa: float = hparams.prior_kappa
        self.niw_nu: float = hparams.NIW_prior_nu

    def init_priors(self, codes:Tensor):
        if self.prior_mu_0_choice == "data_mean":
            self.niw_m = codes.mean(axis=0)
        if self.prior_sigma_choice == "isotropic":
            self.niw_psi = (torch.eye(codes.shape[1]) *
                            self.prior_sigma_scale).double()
        elif self.prior_sigma_choice == "data_std":
            self.niw_psi = (torch.diag(codes.std(axis=0)) *
                            self.prior_sigma_scale).double()
        else:
            raise NotImplementedError()
        return self.niw_m, self.niw_psi

    def compute_params_post(self, codes_k: Tensor, mu_k: Tensor):
        return NIW_compute_params_post(kappa=self.niw_kappa, nu=self.niw_nu, mu_0=self.niw_m, psi=self.niw_psi, codes_k=codes_k, mu_k=mu_k)

    def compute_post_mus(self, N_ks: Tensor, data_mus: Tensor):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        return NIW_compute_post_mus(self.niw_m, self.niw_kappa, N_ks.reshape(-1, 1), data_mus)

    def compute_post_cov(self, N_k: int, mu_k: Tensor, data_cov_k: Tensor):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        if N_k > 0:
            return NIW_compute_post_cov(kappa=self.niw_kappa, nu=self.niw_nu, psi=self.niw_psi, mu_0=self.niw_m, N_k=N_k, mu_k=mu_k, data_cov_k=data_cov_k)
        else:
            return self.niw_psi

    def log_marginal_likelihood(self, codes_k, mu_k):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(
            codes_k, mu_k)
        (N_k, D) = codes_k.shape
        return NIW_log_marginal_likelihood(D=D, N_k=N_k, nu=self.niw_nu, nu_star=nu_star, psi=self.niw_psi, psi_star=psi_star, kappa=self.niw_kappa, kappa_star=kappa_star)


class NIG_prior:
    """A class used to store nig parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    The NIG will model each codes channel separetly, so we will have d-dimensions for every hyperparam
    """

    def __init__(self, hparams, codes_dim):
        self.name = "NIG"
        self.dim = codes_dim
        self.prior_mu_0_choice = hparams.prior_mu_0
        self.nig_V = torch.ones(self.dim) / hparams.prior_kappa
        self.nig_a = torch.ones(self.dim) * (hparams.NIW_prior_nu / 2.0)
        self.prior_sigma_choice = hparams.prior_sigma_choice
        if self.prior_sigma_choice == "iso_005":
            self.nig_sigma_sq_0 = torch.ones(self.dim) * 0.005
        if self.prior_sigma_choice == "iso_0001":
            self.nig_sigma_sq_0 = torch.ones(self.dim) * 0.0001

        self.nig_b = torch.ones(
            self.dim) * (hparams.NIW_prior_nu * self.nig_sigma_sq_0 / 2.0)

    def init_priors(self, codes):
        if self.prior_mu_0_choice == "data_mean":
            self.nig_m = codes.mean(axis=0)
        return self.nig_m, torch.eye(codes.shape[1]) * self.nig_sigma_sq_0

    def compute_params_post(self, codes_k, mu_k=None):
        N = len(codes_k)

        V_star = self.nig_V * (1. / (1 + self.nig_V * N))
        m_star = V_star * (self.nig_m / self.nig_V + codes_k.sum(axis=0))
        a_star = self.nig_a + N / 2.
        b_star = self.nig_b + 0.5 * ((self.nig_m**2) / self.nig_V +
                                     (codes_k**2).sum(axis=0) -
                                     (m_star**2) / V_star)
        return V_star, m_star, a_star, b_star

    def compute_post_mus(self, N_ks, data_mus):
        # kappa = 1.0 / self.nig_V
        # return ((N_ks.reshape(-1, 1) * data_mus) + (kappa * self.nig_m)) / (
        #     N_ks.reshape(-1, 1) + kappa
        # )

        # for each K we are going to have mu in R^D
        return ((N_ks.reshape(-1, 1) * data_mus) +
                (1 / self.nig_V * self.nig_m)) / (N_ks.reshape(-1, 1) +
                                                  1 / self.nig_V)

    def compute_post_cov(self, N_ks, mus, data_stds):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)

        # N_ks is a d-dimentionl tensor wirh N_ks[d] = N_k of the above
        # data_std is a d-dimentional tensor with data_std[d] is the weighted std along dimention d
        if N_ks > 0:
            post_sigma_sq = (
                data_stds * N_ks + 2 * self.nig_b + 1 / self.nig_V *
                ((self.nig_m - mus)**2)) / (N_ks + 2 * self.nig_a + 3)
            return post_sigma_sq
        else:
            return torch.eye(mus.shape[1]) * self.nig_sigma_sq_0

    def log_marginal_likelihood(self, codes_k, mu_k):
        # Hard assignment
        # Since we consider the channels to be independent, the log likelihood will be the sum of log likelihood per channel
        V_star, m_star, a_star, b_star = self.compute_params_post(
            codes_k, mu_k)
        N = len(codes_k)
        lm_ll = 0
        for d in range(self.dim):
            lm_ll += 0.5 * (torch.log(torch.abs(V_star[d])) - torch.log(torch.abs(self.nig_V[d]))) \
                + self.nig_a[d] * torch.log(self.nig_b[d]) - a_star[d] * torch.log(b_star[d]) \
                + lgamma(a_star[d]) - lgamma(self.nig_a[d]) - (N / 2.) * \
                torch.log(torch.tensor(np.pi)) - N * \
                torch.log(torch.tensor(2.))
        return lm_ll

#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import wandb
import argparse
from argparse import ArgumentParser, Namespace
import os
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import lightning.pytorch as pl
from lightning.fabric.utilities import seed
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
import torch

from src.datasets import CustomDataset
from src.datasets import GMM_dataset
from src.clusternet_models.clusternetasmodel import ClusterNetModel
from src.utils import check_args, cluster_acc

from typing import Optional
import logging

logging.getLogger("lightning").setLevel(logging.INFO)


def get_dataset_directory(dataset_name: str) -> str:
    if dataset_name == 'CNG':
        return "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/cng_features"
    elif dataset_name=='tomo':
        return "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/tomo_features"
    elif dataset_name=='proteasome-12':
        return "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/proteasome12_features"
    else:
        raise NotImplementedError


def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dataset", default="custom")
    # Training parameters
    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="input batch size for training")
    parser.add_argument("--num_workers",
                        type=int,
                        default=3,
                        help="num_workers for Dataloader")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=500,
    )
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument("--use_labels_for_eval",
                        action="store_true",
                        help="whether to use labels for evaluation")
    parser.add_argument("--gpus", nargs='*', type=int, default=None)

    # added arguments
    parser.add_argument("--project",
                        type=str,
                        default="DeepDPM",
                        help="wandb project name")

    return parser


def run_on_embeddings_hyperparams(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--init_k",
                        default=1,
                        type=int,
                        help="number of initial clusters")
    parser.add_argument(
        "--clusternet_hidden_layer_list",
        type=int,
        nargs="+",
        default=[50, 50],
        help="The hidden layers in the clusternet. Defaults to [50, 50].",
    )
    parser.add_argument(
        "--transform_input_data",
        type=str,
        default="normalize",
        choices=[
            "normalize", "min_max", "standard", "standard_normalize", "None",
            None
        ],
        help="Use normalization for embedded data",
    )
    parser.add_argument(
        "--cluster_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--how_to_compute_mu",
        type=str,
        choices=["kmeans", "soft_assign"],
        default="soft_assign",
    )
    parser.add_argument(
        "--how_to_init_mu",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans",
    )
    parser.add_argument(
        "--how_to_init_mu_sub",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans_1d",
    )
    parser.add_argument(
        "--cluster_lr",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--subcluster_lr",
        type=float,
        default=0.005,
    )
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="StepLR",
                        choices=["StepLR", "None", "ReduceOnP","CosineAnnealingLR"])
    parser.add_argument(
        "--start_sub_clustering",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--subcluster_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--start_splitting",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--subcluster_softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--split_prob",
        type=float,
        default=None,
        help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--merge_prob",
        type=float,
        default=None,
        help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--init_new_weights",
        type=str,
        default="same",
        choices=["same", "random", "subclusters"],
        help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
    )
    parser.add_argument(
        "--start_merging",
        type=int,
        default=55,
        help="The epoch in which to start consider merge proposals",
    )
    parser.add_argument(
        "--merge_init_weights_sub",
        type=str,
        default="highest_ll",
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_init_weights_sub",
        type=str,
        default="random",
        choices=["same_w_noise", "same", "random"],
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_merge_every_n_epochs",
        type=int,
        default=30,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--raise_merge_proposals",
        type=str,
        default="brute_force_NN",
        help="how to raise merge proposals",
    )
    parser.add_argument(
        "--cov_const",
        type=float,
        default=0.005,
        help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
    )
    parser.add_argument(
        "--freeze_mus_submus_after_splitmerge",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
    )
    parser.add_argument(
        "--freeze_mus_after_init",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
    )
    parser.add_argument(
        "--use_priors",
        type=int,
        default=1,
        help="Whether to use priors when computing model's parameters",
    )
    parser.add_argument("--prior",
                        type=str,
                        default="NIW",
                        choices=["NIW", "NIG"])
    parser.add_argument("--pi_prior",
                        type=str,
                        default="uniform",
                        choices=["uniform", None])
    parser.add_argument(
        "--prior_dir_counts",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prior_kappa",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--NIW_prior_nu",
        type=float,
        default=None,
        help="Need to be at least codes_dim + 1",
    )
    parser.add_argument(
        "--prior_mu_0",
        type=str,
        default="data_mean",
    )
    parser.add_argument(
        "--prior_sigma_choice",
        type=str,
        default="isotropic",
        choices=["iso_005", "iso_001", "iso_0001", "data_std"],
    )
    parser.add_argument(
        "--prior_sigma_scale",
        type=float,
        default=".005",
    )
    parser.add_argument(
        "--compute_params_every",
        type=int,
        help="How frequently to compute the clustering params (mus, sub, pis)",
        default=1,
    )
    parser.add_argument(
        "--start_computing_params",
        type=int,
        help="When to start to compute the clustering params (mus, sub, pis)",
        default=25,
    )
    parser.add_argument(
        "--cluster_loss",
        type=str,
        help="What kind og loss to use",
        default="KL_GMM_2",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )
    parser.add_argument(
        "--subcluster_loss",
        type=str,
        help="What kind og loss to use",
        default="isotropic",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )

    parser.add_argument(
        "--log_metrics_at_train",
        type=bool,
        default=True,
    )
    parser.add_argument("--evaluate_every_n_epochs",
                        type=int,
                        default=5,
                        help="How often to evaluate the net")
    return parser


def need_resume(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        print(f"No such file or directory: {checkpoint_dir}. Skip resuming")
        return None
    files = [
        f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')
        and os.path.isfile(os.path.join(checkpoint_dir, f))
    ]

    if len(files) == 0:
        print(f"no checkpoint file found in {checkpoint_dir}. Skip resume")
        return None

    files = [f for f in files if "end" not in f]

    if len(files) == 0:
        print(
            f"no checkpoint file except the last model found in {checkpoint_dir}. Skip resume"
        )
        return None

    checkpoint_path = os.path.join(checkpoint_dir, files[0])

    return os.path.abspath(checkpoint_path)


def get_checkpoint_callback(args: Namespace):
    if args.save_checkpoints:
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./saved_models/{args.dataset}/{args.exp_name}",
            every_n_epochs=10)
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        if not os.path.exists(
                f'./saved_models/{args.dataset}/{args.exp_name}'):
            os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')
    else:
        checkpoint_callback = None

    return checkpoint_callback


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Only_for_embbedding")
    parser = parse_minimal_args(parser)
    parser = run_on_embeddings_hyperparams(parser)
    args = parser.parse_args()
    args.train_cluster_net = args.max_epochs
    args.dir = get_dataset_directory(args.dataset)
    return args


def train_cluster_net():

    args = get_args()
    dataset_obj = GMM_dataset(
        args) if args.dataset == "synthetic" else CustomDataset(args)

    train_loader, val_loader = dataset_obj.get_loaders()

    check_args(args, dataset_obj.data_dim)
    wandb.init(project=args.project,
               name=args.exp_name,
               sync_tensorboard=True,
               mode='offline',
               config=args)
    logger = TensorBoardLogger(save_dir="tensorboard", name=args.project)

    seed.seed_everything(args.seed)

    resume_path = need_resume(f"./saved_models/{args.dataset}/{args.exp_name}")
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=torch.device('cpu'))
        # make things easier
        args.init_k = ckpt['K']
        print(f"override init_k to {args.init_k}")

    model = ClusterNetModel(hparams=args,
                            input_dim=dataset_obj.data_dim,
                            init_k=args.init_k)
    # if accelerator=ddp, loading from checkpoint has a weird bug on which I have no ideas
    trainer = pl.Trainer(logger=logger,
                         max_epochs=args.max_epochs,
                         devices=args.gpus,
                         num_sanity_val_steps=0,
                         callbacks=get_checkpoint_callback(args),
                         enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    print("Finished training!")
    # evaluate last model
    dataset = dataset_obj.get_train_data()
    data = dataset.data
    net_pred = model(data).argmax(axis=1).cpu().numpy()
    if args.use_labels_for_eval:
        # evaluate model using labels
        labels = dataset.targets.numpy()
        acc = np.round(cluster_acc(labels, net_pred), 5)
        nmi = np.round(NMI(net_pred, labels), 5)
        ari = np.round(ARI(net_pred, labels), 5)

        print(
            f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(net_pred))}"
        )

    return net_pred


if __name__ == "__main__":
    train_cluster_net()

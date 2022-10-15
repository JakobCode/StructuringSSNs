# Parts of this script are taken from:https://github.com/MiguelMonteiro/PHiSeg-code.
#
# The source repository is under Apache-2.0 License.
# Authors from original repository: Miguel Monteiro
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous


import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
import yaml
from LIDC.data_handling.lidc_data import lidc_data as data_loader
from LIDC.data_handling.utils import (
    generalised_energy_distance,
    list_mean,
    to_one_hot,
    variance_ncc_dist,
)
from LIDC.models.phiseg_ssn import StochasticSegmentationNetwork
from medpy.metric import dc
from utils.loss import ssn_loss_CE

#############


# define and parse arguments
parser = argparse.ArgumentParser()

# config
parser.add_argument(
    "--experiment_folder",
    type=str,
    help="experiment root folder")

parser.add_argument(
    "--model_type",
    type=str,
    default="model_best_loss",
    help="type of stored model to be loaded.",
)

args = parser.parse_args()

exp_folder = args.experiment_folder
cfg_path = os.path.join(exp_folder, "config.yml")

with open(cfg_path, "r") as f:
    try:
        cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# Experiments Configuration
model_save_dir = os.path.join(exp_folder, "saved_models")
seed = cfg["experiment"]["seed"]

# Data Configuration
data_root = cfg["data"]["data_file"]
preproc_folder = cfg["data"]["data_folder_preproc"]
annotator_range = range(cfg["data"]["num_annotators"])
nlabels = cfg["data"]["num_labels"]
img_shape = cfg["data"]["shape"]

# SSN Configuration
rank = cfg["ssn"]["rank"]


model_save_dir = os.path.join(exp_folder, "saved_models")
result_folder = os.path.join(exp_folder, "results")
os.makedirs(result_folder, exist_ok=True)


checkpoint_file = os.path.join(
    exp_folder,
    "saved_models",
    args.model_type +
    ".ckpt")

target_folder_pkl = os.path.join(result_folder, "pkls")
os.makedirs(target_folder_pkl, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

# data loader
data = data_loader(
    annotator_range=annotator_range,
    data_root=data_root,
    preproc_folder=preproc_folder)

# model
model = StochasticSegmentationNetwork(
    resolution_levels=7,
    n_classes=nlabels,
    rank=rank,
    img_size=img_shape,
    n0=32)

# loss
criterion = ssn_loss_CE()

checkpoint = torch.load(checkpoint_file, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

with torch.no_grad():

    dice_list = []
    loss_list = []
    ged_list = []
    ncc_list = []

    num_samples = len(data.test.images)

    for sample_step in range(num_samples):

        ii = sample_step

        print(f"Process sample {sample_step} of {num_samples}")

        x = data.test.images[ii, ...]
        s_gt_arr = data.test.labels[ii, ...]
        s = s_gt_arr[:, :, np.random.choice(annotator_range)]

        # 1 x 1 x 128 x 128
        x_b = (
            torch.tensor(x, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # 16 x 128 x 128
        s_b = s  # np.tile(s, [16, 1, 1])

        # 1 x 1 x 128 x 128
        y_pred = model(x_b)

        # save prediction information to pickle file
        with open(
            os.path.join(target_folder_pkl, f"phiseg_run_res_sample_{ii}.pkl"), "wb"
        ) as f:
            pkl.dump(
                {
                    "sample_id": id,
                    "data": "LIDC",
                    "rank": model.rank,
                    "model_path": checkpoint_file,
                    "num_classes": 2,
                    "dim": [1, 128, 128],
                    "mean": model.last_mean[0],
                    "cov_diag": model.last_cov_diag[0],
                    # "cov_logdiag": model.last_log_cov_diag[0],
                    "cov_factor": model.last_cov_factor[0],
                    "sample": x_b[0].detach().cpu().numpy(),
                    "labels": s_gt_arr,
                    "annotations": 1,
                },
                f,
            )

        loss_list.append(
            criterion(
                torch.tensor(
                    s_b,
                    dtype=torch.long,
                    device=device),
                y_pred) .detach() .cpu() .numpy())

        s_pred_sm_arr = (
            torch.softmax(y_pred.reshape((-1, 2, 128, 128)), dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        s_pred_sm_arr = np.transpose(s_pred_sm_arr, (0, 2, 3, 1))
        s_pred_sm_mean_ = s_pred_sm_arr.mean(axis=0)

        s_pred_arr = np.argmax(s_pred_sm_arr, axis=-1)

        s_gt_arr_r = s_gt_arr.transpose((2, 0, 1))  # num gts x X x Y

        s_gt_arr_r_sm = to_one_hot(
            s_gt_arr_r, nlabels=nlabels
        )  # num gts x X x Y x nlabels

        ged, diversity = generalised_energy_distance(
            s_pred_arr, s_gt_arr_r, nlabels=nlabels - 1, label_range=range(1, nlabels)
        )

        if s_pred_sm_arr.shape[-1] == 1:
            s_pred_sm_arr = np.concatenate(
                [1 - s_pred_sm_arr, s_pred_sm_arr], axis=-1)
        ncc = variance_ncc_dist(s_pred_sm_arr, s_gt_arr_r_sm)

        s_ = np.argmax(s_pred_sm_mean_, axis=-1)

        # Write losses to list
        per_lbl_dice = []
        for lbl in range(nlabels):
            binary_pred = (s_ == lbl) * 1
            binary_gt = (s_gt_arr_r == lbl) * 1

            for i in range(binary_gt.shape[0]):

                # no pos. class, all correct
                if np.sum(binary_gt[i]) == 0 and np.sum(binary_pred) == 0:
                    per_lbl_dice.append(1)

                # no positive class but not predicted correct
                elif (
                    np.sum(binary_pred) > 0
                    and np.sum(binary_gt[i]) == 0
                    or np.sum(binary_pred) == 0
                    and np.sum(binary_gt) > 0
                ):
                    per_lbl_dice.append(0)
                else:
                    per_lbl_dice.append(dc(binary_pred, binary_gt[i]))

        dice_list.append(per_lbl_dice)
        ged_list.append(ged)
        ncc_list.append(ncc)

    dice_arr = np.asarray(dice_list)
    per_structure_dice = dice_arr.mean(axis=0)

    avg_loss = np.mean(loss_list)
    avg_ged = list_mean(ged_list)
    avg_ncc = list_mean(ncc_list)

    print(f"     Performance: ")
    print(f"          loss:        {avg_loss}")
    print(f"          ncc:         {np.mean(avg_ncc)}")
    print(f"          ged:         {avg_ged}")
    print(f"          dice:        {np.mean(per_structure_dice)}")
    print(f"          diversity:   {diversity}")

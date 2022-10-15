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
import random
from datetime import datetime

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
from LIDC.models.phiseg_ssn import StochasticSegmentationNetwork as SSN
from medpy.metric import dc
from utils.loss import ssn_loss_CE

#############

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", metavar="-cfg", type=str, help="Configuration yaml file."
)
args = parser.parse_args()

print(args)
cfg_path = args.config
assert os.path.exists(
    cfg_path), "Configuration file '{}' does not exist!".format(cfg_path)
assert cfg_path.endswith(".yaml") or cfg_path.endswith(
    ".yml"
), "Configuration file should end with '.yaml' or '.yml'."

with open(cfg_path, "r") as f:
    try:
        cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


print(" ##################   EXPERIMENT SETUP   ################## ")
print(cfg)
print(" ########################################################## ")

# Experiments Configuration
exp_name = (
    cfg["experiment"]["experiment_name"]
    + "_"
    + str(datetime.now())
    .replace("-", "")
    .replace(":", "")
    .replace(".", "")
    .replace(" ", "")
)
exp_folder = os.path.join(cfg["experiment"]["experiment_folder"], exp_name)
model_save_dir = os.path.join(exp_folder, "saved_models")
seed = cfg["experiment"]["seed"]

os.makedirs(exp_folder, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# save config file to experiment folder
with open(os.path.join(exp_folder, "config.yml"), "w") as f:
    documents = yaml.dump(cfg, f)

# Data Configuration
data_pkl_file = cfg["data"]["data_file"]
preproc_folder = cfg["data"]["data_folder_preproc"]
annotator_range = range(cfg["data"]["num_annotators"])
nlabels = cfg["data"]["num_labels"]
img_shape = cfg["data"]["shape"]

# SSN Configuration
rank = cfg["ssn"]["rank"]

# Optimizer Configuration
learning_rate = cfg["optimizer"]["learning_rate"]

# Training Configuration:
train_steps = cfg["training"]["train_steps"]
val_step = cfg["training"]["val_steps"]
batch_size_train = cfg["training"]["batch_size_train"]
batch_size_val = cfg["training"]["batch_size_val"]


# Prepare Training
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Used device: {}".format(device))

# data loader
data = data_loader(
    annotator_range=annotator_range,
    data_root=data_pkl_file,
    preproc_folder=preproc_folder)

# model
model = SSN(
    resolution_levels=7,
    n_classes=nlabels,
    rank=rank,
    img_size=img_shape,
    n0=32)

# loss
criterion = ssn_loss_CE()

# training
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
model.to(device)

best_ncc = -1
best_dice = -1
best_loss = 999999999
best_ged = 999999999

# Start Training
for iter in range(train_steps):

    model.train()
    x_b, s_b = data.train.next_batch(batch_size_train)

    x_b = torch.tensor(x_b, device=device,
                       dtype=torch.float32).permute([0, 3, 1, 2])
    s_b = torch.tensor(s_b, dtype=torch.long, device=device)

    optimizer.zero_grad()
    y_pred = model(x_b)
    loss = criterion(s_b, y_pred)

    loss.backward()
    optimizer.step()

    print(
        "Step {} of {}: {}".format(
            iter,
            train_steps,
            loss.detach().cpu().numpy()))
    if (iter + 1) % val_step == 0:
        model.eval()

        with torch.no_grad():
            print("   Start Validation")

            num_batches = 0

            dice_list = []
            loss_list = []
            ged_list = []
            ncc_list = []

            N = 100
            for ii in range(N):

                x = data.validation.images[ii, ...]
                s_gt_arr = data.validation.labels[ii, ...]
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
                    s_pred_arr,
                    s_gt_arr_r,
                    nlabels=nlabels - 1,
                    label_range=range(1, nlabels),
                )

                if s_pred_sm_arr.shape[-1] == 1:
                    s_pred_sm_arr = np.concatenate(
                        [1 - s_pred_sm_arr, s_pred_sm_arr], axis=-1
                    )
                ncc = variance_ncc_dist(s_pred_sm_arr, s_gt_arr_r_sm)

                s_ = np.argmax(s_pred_sm_mean_, axis=-1)

                # Write losses to list
                per_lbl_dice = []
                for lbl in range(nlabels):
                    binary_pred = (s_ == lbl) * 1
                    binary_gt = (s_gt_arr_r == lbl) * 1

                    # no pos. class, all correct
                    if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0:
                        per_lbl_dice.append(1)

                    # no positive class but not predicted correct
                    elif (
                        np.sum(binary_pred) > 0
                        and np.sum(binary_gt) == 0
                        or np.sum(binary_pred) == 0
                        and np.sum(binary_gt) > 0
                    ):
                        per_lbl_dice.append(0)
                    else:
                        per_lbl_dice.append(dc(binary_pred, binary_gt))

                num_batches += 1

                dice_list.append(per_lbl_dice)
                ged_list.append(ged)
                ncc_list.append(ncc)

            dice_arr = np.asarray(dice_list)
            per_structure_dice = dice_arr.mean(axis=0)

            avg_loss = np.mean(loss_list)
            avg_ged = list_mean(ged_list)
            avg_ncc = np.mean(ncc_list)

            # save models
            if np.mean(per_structure_dice) > best_dice:
                print(
                    f"best_dice improved from {best_dice} to {np.mean(per_structure_dice) }!"
                )
                best_dice = np.mean(per_structure_dice)
                best_file = os.path.join(
                    model_save_dir, "model_best_dice.ckpt")
                torch.save(
                    {
                        "step": iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dice_val": best_dice,
                    },
                    best_file,
                )

            if avg_loss < best_loss:
                print(f"best_loss improved from {best_loss} to {avg_loss}!")
                best_loss = np.mean(avg_loss)
                best_file = os.path.join(
                    model_save_dir, "model_best_loss.ckpt")
                torch.save(
                    {
                        "step": iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    best_file,
                )

            if avg_ged < best_ged:
                print(f"best_ged improved from {best_ged} to {avg_ged}!")
                best_ged = avg_ged
                best_file = os.path.join(model_save_dir, "model_best_ged.ckpt")
                torch.save(
                    {
                        "step": iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_ged": best_ged,
                    },
                    best_file,
                )

            if avg_ncc > best_ncc:
                print(f"best_ncc improved from {best_ncc} to {avg_ncc}!")
                best_ncc = avg_ncc
                best_file = os.path.join(model_save_dir, "model_best_ncc.ckpt")
                torch.save(
                    {
                        "step": iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_ncc": best_ncc,
                    },
                    best_file,
                )

            print("     Validation: ")
            print("          loss:        {} / {}".format(avg_loss, best_loss))
            print("          ncc:         {} / {}".format(avg_ncc, best_ncc))
            print("          ged:         {} / {}".format(avg_ged, best_ged))
            print(
                "          dice:        {} / {}".format(
                    np.mean(per_structure_dice), best_dice
                )
            )
            print("          diversity:   {}".format(diversity))

best_file = os.path.join(model_save_dir, "final_model.ckpt")
torch.save(
    {
        "step": iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    best_file,
)

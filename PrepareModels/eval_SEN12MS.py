# Parts of this script are taken from: https://github.com/schmitt-muc/SEN12MS.
#
# The source repository is under Custom Lisence: https://github.com/schmitt-muc/SEN12MS/blob/master/LICENSE.txt
# Authors from original repository:
# Prof. Michael Schmitt, michael.schmitt@unibw.de, +49 89 6004-4426, https://www.unibw.de/lrt9/lrt-9.3/.
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

import argparse
import os
import pickle as pkl
import random

import matplotlib.pyplot as plt
import numpy as np
import SEN12MS.metrics as metrics
import torch
import torchvision.transforms as T
import yaml
from SEN12MS.datasets import SEN12MS
from SEN12MS.models import StochasticSegmentationNetwork as SSN

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
    default="model_best_aa",
    help="type of stored model to be loaded. Default: model_best_aa",
)

args = parser.parse_args()

exp_folder = args.experiment_folder
model_save_dir = os.path.join(exp_folder, "saved_models")
result_folder = os.path.join(exp_folder, "results")
os.makedirs(result_folder, exist_ok=True)

cfg_path = os.path.join(exp_folder, "config.yml")

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


seed = cfg["experiment"]["seed"]
data_dir = cfg["data"]["data_root"]
nlabels = cfg["data"]["num_labels"]
img_shape = cfg["data"]["shape"]

# SSN Configuration
rank = cfg["ssn"]["rank"]

checkpoint_file = os.path.join(
    exp_folder,
    "saved_models",
    args.model_type +
    ".ckpt")

target_folder_pkl = os.path.join(result_folder, "pkls")
os.makedirs(target_folder_pkl, exist_ok=True)

# set flags for GPU processing if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset
print("SSN hold out data set")
dataset = SEN12MS(
    data_dir,
    subset="ssn_holdout",
    no_savanna=False,
    use_s2hr=True,
    use_s2mr=True,
    use_s2lr=True,
    use_s1=False,
)
gt_id = "lc"

n_classes = dataset.n_classes
n_inputs = dataset.n_inputs

# set up network
model = SSN(n_classes=n_classes, n_channels=n_inputs, rank=rank)

model.to(device)

# restore network weights
state = torch.load(checkpoint_file, map_location=device)


step = state["step"]
model.load_state_dict(state["model_state_dict"])
model.eval()
print("loaded checkpoint from step", step)

# initialize scoring if ground-truth is available
conf_mat = metrics.ConfMatrix(n_classes)


# predict samples
for n in np.arange(len(dataset)):
    print("Step {} of {}".format(n, len(dataset)))
    sample = dataset[n]

    # unpack sample
    image = torch.Tensor(sample["image"]).unsqueeze(0)
    target = torch.Tensor(sample["label"]).unsqueeze(0)

    # move data to gpu if model is on gpu
    image = image.to(device)
    target = target.to(device)

    # forward pass
    with torch.no_grad():
        prediction = model(image)

    # convert to 256x256 numpy arrays
    prediction = prediction.cpu().numpy()
    prediction = np.argmax(prediction, axis=-3)
    means = np.reshape(model.last_mean, [len(prediction), 256, 256, n_classes])
    target = target.cpu().numpy()

    # save predicted distribution
    id = sample["id"].replace("_s2_", "_" + gt_id + "_")
    with open(
        os.path.join(target_folder_pkl, f"sen12ms_run_res_sample_{id}.pkl"), "wb"
    ) as f:
        pkl.dump(
            {
                "sample_id": id,
                "data": "SEN12MS",
                "rank": model.rank,
                "model_path": checkpoint_file,
                "num_classes": n_classes,
                "dim": image[0].shape,
                "mean": model.last_mean[0],
                "cov_diag": model.last_cov_diag[0],
                "cov_logdiag": model.last_log_cov_diag[0],
                "cov_factor": model.last_cov_fac[0],
                "sample": image[0].detach().cpu().numpy(),
                "labels": [target[0]],
                "annotations": 1,
            },
            f,
        )

    # update error metrics
    conf_mat.add(target[0], np.argmax(means, -1)[0])

# print scoring results
if args.score:
    print("AA\t", conf_mat.get_aa())
    print("mIoU\t", conf_mat.get_mIoU())

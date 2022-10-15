# Parts of this script are taken from: https://github.com/bfortuner/pytorch_tiramisu.
#
# The source repository is under MIT License.
# Authors from original repository: Brendan Fortuner
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

#############
import argparse
import os

import CamVid.models.tiramisu as tiramisu
import CamVid.utils.training as train_utils
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from CamVid.datasets import camvid, joint_transforms
from utils.loss import ssn_loss_CE

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
    default="model_best_iou",
    help="type of stored model to be loaded. Default: model_best_iou",
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
batch_size = cfg["training"]["batch_size_test"]

# SSN Configuration
rank = cfg["ssn"]["rank"]

checkpoint_file = os.path.join(
    exp_folder,
    "saved_models",
    args.model_type +
    ".cpkt")
CAMVID_PATH = data_dir
PKL_PATH = os.path.join(result_folder, "pkls")
os.makedirs(PKL_PATH, exist_ok=True)


normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)

test_dset = camvid.CamVid(
    CAMVID_PATH,
    "test",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False
)

print("Test: %d" % len(test_loader.dataset.imgs))
print("Classes: %d" % len(test_loader.dataset.classes))

inputs, targets = next(iter(test_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

device = "cuda" if torch.cuda.is_available() else "cpu"


model = tiramisu.FCDenseNetSSN103(n_classes=11).to(device)
criterion = ssn_loss_CE(
    weight=camvid.class_weight[:-1].to(device), ignore_index=11)


train_utils.load_weights(fpath=checkpoint_file, model=model)


with torch.no_grad():

    torch.cuda.manual_seed(0)
    train_utils.view_sample_predictions(
        model=model,
        loader=test_loader,
        save_path=None,
        pkl_path=PKL_PATH,
        save_pkl=True,
    )

    torch.cuda.manual_seed(0)
    print("Mean prediction resutls: ")
    print(
        "loss: {}, error: {}, iou: {}".format(
            *train_utils.test(model, test_loader, criterion)
        )
    )

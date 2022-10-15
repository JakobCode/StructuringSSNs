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
import random
from datetime import datetime

import CamVid.utils.training as train_utils
import numpy as np
import torch
import yaml
from CamVid.datasets import camvid, joint_transforms
from CamVid.models import tiramisu
from torchvision import transforms
from utils.loss import ssn_loss_CE

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", metavar="--cfg", type=str, help="Configuration yaml file."
)
args = parser.parse_args()

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
data_root = cfg["data"]["data_root"]
annotator_range = range(cfg["data"]["num_annotators"])
nlabels = cfg["data"]["num_labels"]
img_shape = cfg["data"]["shape"]

# SSN Configuration
rank = cfg["ssn"]["rank"]

# Optimizer Configuration
learning_rate = cfg["optimizer"]["learning_rate"]
momentum = cfg["optimizer"]["momentum"]
weight_decay = cfg["optimizer"]["weight_decay"]
lr_decay = cfg["optimizer"]["lr_decay"]
decay_every_n_epochs = cfg["optimizer"]["decay_every_n_epochs"]


# Training Configuration:
num_epochs = cfg["training"]["num_epochs"]
val_step = cfg["training"]["val_steps"]
batch_size_train = cfg["training"]["batch_size_train"]
batch_size_val = cfg["training"]["batch_size_val"]
batch_size_test = cfg["training"]["batch_size_test"]

# Prepare Training
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Used device: {}".format(device))


normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose(
    [
        joint_transforms.JointRandomHorizontalFlip()    
    ]
)

train_dset = camvid.CamVid(
    data_root,
    "train",
    joint_transform=train_joint_transformer,
    transform=transforms.Compose([transforms.ToTensor(), normalize, ]),
)
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size_train, shuffle=True
)

val_dset = camvid.CamVid(
    data_root,
    "val",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size_val, shuffle=False
)

test_dset = camvid.CamVid(
    data_root,
    "test",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size_test, shuffle=False
)


print("Train: %d" % len(train_loader.dataset.imgs))
print("Val: %d" % len(val_loader.dataset.imgs))
print("Test: %d" % len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())


device = "cuda" if torch.cuda.is_available() else "cpu"

# ignore class 12 in network and loss
model = tiramisu.FCDenseNetSSN103(n_classes=nlabels, rank=rank).to(device)
criterion = ssn_loss_CE(
    weight=camvid.class_weight[:-1].to(device), ignore_index=11)

model.apply(train_utils.weights_init)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

best_iou = 0
last_improve = 0

for epoch in range(1, num_epochs + 1):
    since = datetime.now()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch
    )
    print(
        "Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, trn_loss, 1 - trn_err
        )
    )
    time_elapsed = (datetime.now() - since).total_seconds()
    print(
        "Train Time {:.0f}m {:.0f}s".format(
            time_elapsed //
            60,
            time_elapsed %
            60))

    ### Test ###
    val_loss, val_err, iou = train_utils.test(
        model, val_loader, criterion, epoch)
    print(
        "Val - Loss: {:.4f} | Acc: {:.4f} | IoU: {:.4f}".format(
            val_loss, 1 - val_err, iou
        )
    )
    time_elapsed = (datetime.now() - since).total_seconds()
    print(
        "Total Time {:.0f}m {:.0f}s\n".format(
            time_elapsed //
            60,
            time_elapsed %
            60))

    ### Checkpoint ###
    if iou > best_iou:
        print("iou improved from ", best_iou, " to ", iou)
        train_utils.save_weights(
            model_save_dir, "model_best_iou", model, epoch, val_loss, val_err, iou
        )
        best_iou = iou

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(
        learning_rate, lr_decay, optimizer, epoch, decay_every_n_epochs
    )

    if epoch - last_improve == 100:
        print("Stop training after ", epoch, " epochs!")
        break

train_utils.test(model, test_loader, criterion, epoch=1)

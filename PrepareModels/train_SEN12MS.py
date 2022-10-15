# Parts of this script are taken from: https://github.com/schmitt-muc/SEN12MS.
#
# The source repository is under Custom Lisence: https://github.com/schmitt-muc/SEN12MS/blob/master/LICENSE.txt
# Authors from original repository:
# Prof. Michael Schmitt, michael.schmitt@unibw.de, +49 89 6004-4426, https://www.unibw.de/lrt9/lrt-9.3/.
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

#############
import argparse
import os
import pickle as pkl
import random
from datetime import datetime

import numpy as np
import SEN12MS.metrics as metrics
import torch
import yaml
from SEN12MS.datasets import SEN12MS

from SEN12MS.models import StochasticSegmentationNetwork as SSN
from torch.utils.data import DataLoader
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
nlabels = cfg["data"]["num_labels"]
img_shape = cfg["data"]["shape"]

# SSN Configuration
rank = cfg["ssn"]["rank"]

# Optimizer Configuration
learning_rate = cfg["optimizer"]["learning_rate"]
momentum = cfg["optimizer"]["momentum"]
weight_decay = cfg["optimizer"]["weight_decay"]

# Training Configuration:
num_epochs = cfg["training"]["num_epochs"]
val_step = cfg["training"]["val_steps"]
batch_size_train = cfg["training"]["batch_size_train"]
batch_size_val = cfg["training"]["batch_size_val"]
num_workers = cfg["training"]["num_workers"]

# Prepare Training
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Used device: {}".format(device))

# load datasets
train_set = SEN12MS(
    data_root,
    subset="train",
    no_savanna=False,
    use_s2hr=True,
    use_s2mr=True,
    use_s2lr=True,
    use_s1=False,
)

n_classes = train_set.n_classes
n_inputs = train_set.n_inputs

val_set = SEN12MS(
    data_root,
    subset="holdout",
    no_savanna=False,
    use_s2hr=True,
    use_s2mr=True,
    use_s2lr=True,
    use_s1=False,
)

# set up dataloaders
train_loader = DataLoader(
    train_set,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size_val,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)

# set up network
model = SSN(n_classes=n_classes, n_channels=13, rank=rank)
model.to(device)
# run training for one epoch
# set model to train mode
criterion = ssn_loss_CE(ignore_index=255)

# set up optimizer
optimizer = torch.optim.RMSprop(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

# train network
step = 0
best_aa = -1

for epoch in range(num_epochs):
    print("=" * 20, "EPOCH", epoch + 1, "/", str(num_epochs), "=" * 20)

    model.train()

    # main training loop
    for i, batch in enumerate(train_loader):

        # unpack sample
        image, target = batch["image"], batch["label"]

        # reset gradients
        optimizer.zero_grad()

        # move data to gpu if model is on gpu
        image.to(device)
        target.to(device)

        # forward pass
        prediction = model(image)
        loss = criterion(target, prediction)

        # backward pass
        loss.backward()
        optimizer.step()

        # log progress, validate, and save checkpoint
        global_step = i + step

        # run validation
        if global_step > 0 and global_step % val_step == 0:
            # do validation step
            print("Start validation.")
            # set model to evaluation mode
            model.eval()

            with torch.no_grad():
                # main validation loop
                loss = 0
                conf_mat = metrics.ConfMatrix(val_loader.dataset.n_classes)
                for i, batch in enumerate(val_loader):

                    # unpack sample
                    image, target = batch["image"], batch["label"]

                    # move data to gpu if model is on gpu
                    image.to(device)
                    target.to(device)

                    # forward pass
                    with torch.no_grad():
                        prediction = model(image)
                    loss += criterion(target, prediction).cpu().item()

                    # calculate error metrics
                    conf_mat.add_batch(target, model.last_mean.reshape(
                        [-1, 256, 256, 10]).argmax(-1))

                    aa = conf_mat.get_aa() * 100

                if aa > best_aa:
                    print(
                        "best_aa improved from {} to {}!".format(
                            best_aa, aa))
                    best_aa = aa
                    best_file = os.path.join(
                        model_save_dir, "model_best_aa.ckpt")
                    torch.save(
                        {
                            "step": iter,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_aa": best_aa,
                        },
                        best_file,
                    )
                else:
                    print("Validation AA: {}".format(aa))

save_name = os.path.join(model_save_dir, "final_model.ckpt")
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    save_name
)

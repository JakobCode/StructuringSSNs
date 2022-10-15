# Parts of this script are taken from: https://github.com/schmitt-muc/SEN12MS.
#
# The source repository is under Custom Lisence: https://github.com/schmitt-muc/SEN12MS/blob/master/LICENSE.txt
# Authors from original repository:
# Prof. Michael Schmitt, michael.schmitt@unibw.de, +49 89 6004-4426, https://www.unibw.de/lrt9/lrt-9.3/.
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

import glob
import os

import numpy as np
import pandas as pd
import rasterio
import torch.utils.data as data
from tqdm import tqdm

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]

# util function for reading s2 data


def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    return s1


# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    assert no_savanna == False

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # use simplified version of the labels only
    lc = np.take([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10], lc)

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc


# util function for reading data from single sample
def load_sample(
    sample,
    use_s1,
    use_s2hr,
    use_s2mr,
    use_s2lr,
    no_savanna=False,
    igbp=True,
    unlabeled=False,
):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)

    # load s1 data
    if use_s1:
        if use_s2:
            img = np.concatenate((img, load_s1(sample["s1"])), axis=0)
        else:
            img = load_s1(sample["s1"])

    # load label
    if unlabeled:
        return {"image": img, "id": sample["id"]}
    else:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
        return {"image": img, "label": lc, "id": sample["id"]}


# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""

    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def __init__(
        self,
        path,
        subset="train",
        no_savanna=False,
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
    ):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError(
                "No input specified, set at least one of "
                + "use_[s2hr, s2mr, s2lr, s1] to True!"
            )
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        assert subset in ["train", "holdout", "ssn_holdout"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input

        # provide number of classes
        if no_savanna:
            self.n_classes = 10 - 1
            self.no_savanna = True
        else:
            self.n_classes = 10

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []
        if subset == "train":
            pbar = tqdm(total=162556)  # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)  # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(
            pd.read_csv(
                os.path.join(
                    path,
                    "SEN12MS_holdOutScenes.txt"),
                header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]

        val_list_ssn = list(
            pd.read_csv(
                os.path.join(
                    path,
                    "SEN12MS_ssn_holdOutScenes.txt"),
                header=None)[0])
        val_list_ssn = [x.replace("s1_", "s2_") for x in val_list]

        # compile a list of paths to all samples
        if subset == "train":
            train_list = []
            for seasonfolder in [
                "ROIs1970_fall",
                "ROIs1158_spring",
                "ROIs2017_winter",
                "ROIs1868_summer",
            ]:
                train_list += [
                    os.path.join(seasonfolder, x)
                    for x in os.listdir(os.path.join(path, seasonfolder))
                ]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif subset == "holdout":
            sample_dirs = val_list
        elif subset == "ssn_holdout":
            sample_dirs = val_list_ssn

        for folder in sample_dirs:
            s2_locations = glob.glob(
                os.path.join(path, f"{folder}/*.tif"), recursive=True
            )

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer/s2_146":
                broken_file = os.path.join(
                    path,
                    "ROIs1868_summer",
                    "s2_146",
                    "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)
                pbar.write(
                    "ignored one sample because of nan values in " +
                    "the s1 data")
            if folder == "ROIs1158_spring/s2_1":
                broken_file = os.path.join(
                    path, "ROIs1158_spring", "s2_1", "ROIs1158_spring_s2_1_p127.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample to to index error")
                broken_file = os.path.join(
                    path, "ROIs1158_spring", "s2_1", "ROIs1158_spring_s2_1_p132.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample to to index error")
                broken_file = os.path.join(
                    path, "ROIs1158_spring", "s2_1", "ROIs1158_spring_s2_1_p219.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample to to index error")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append(
                    {
                        "lc": lc_loc,
                        "s1": s1_loc,
                        "s2": s2_loc,
                        "id": os.path.basename(s2_loc),
                    }
                )

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        return load_sample(
            sample,
            self.use_s1,
            self.use_s2hr,
            self.use_s2mr,
            self.use_s2lr,
            no_savanna=self.no_savanna,
            igbp=True,
        )

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

"""
The following three scripts process a SSN prediction and are supposed 
to be run one after another: 

1. [This Script] - PreComputeRotations.py
2. EvalRotatedFactors.py
3. PrintResults.py 

This script computes rotatations on a factor model received as a Stochastic Neural Network. 
Loads a single pickle file or all pickle files in a given folder. The rotations
are saved as the rotation matrix within the loaded pickel files. 

Parameters:
--pkl_source   :    single pickle file or folder containing pickle files
--rotation     :    rotations to run. Either 'all' or rotation name
--num_reps     :    number of repetitions of the optimiazation with random initialization
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent))
sys.path.append(
    os.path.join(
        str(Path(os.path.realpath(__file__)).parent.parent), "FactorRotations", "utils"
    )
)

from FactorRotations.utils.factormodel import FactorModel, Rotations
import numpy as np
import traceback
import pickle as pkl
import argparse

# define and parse arguments
parser = argparse.ArgumentParser()

# config
parser.add_argument(
    "--pkl_source",
    type=str,
    help="single pickle file or folder containing pickle files",
)
parser.add_argument(
    "--rotation",
    type=str,
    default="all",
    help="rotations to run. Either 'all' or rotation name",
)
parser.add_argument(
    "--no_reps",
    type=int,
    default=1,
    help="single pickle file or folder containing pickle files",
)
args = parser.parse_args()

pkl_source = args.pkl_source
num_repetitions = args.no_reps
rots_select = args.rotation.lower()

# case 1: single pickle file to process
if pkl_source.endswith(".pkl"):
    pkl_files = [pkl_source]

# case 2: root folder given
else:
    pkl_files = sorted(
        [
            os.path.join(pkl_source, f)
            for f in os.listdir(pkl_source)
            if f.endswith(".pkl")
        ]
    )


rot_dict = {
    "varimax": Rotations.VARIMAX,
    "fpvarimax": Rotations.FPVARIMAX,
    "equamax": Rotations.EQUAMAX,
    "fpequamax": Rotations.FPEQUAMAX,
    "quartimax": Rotations.QUARTIMAX,
    "fpquartimax": Rotations.FPQUARTIMAX,
}

print("#####################################")
if rots_select == "all":
    rotations = list(rot_dict.values())
    print(
        "Compute all {} rotations for {} pickle files".format(
            len(rotations), len(pkl_files)
        )
    )
else:
    rotations = [rot_dict[args.rotation.lower()]]
    print(
        "Compute {} rotation for {} pickle files".format(
            rots_select,
            len(pkl_files)))


print("Start preparation ...")


for i, pkl_file in enumerate(pkl_files):

    print("Process file {} of {}".format(i + 1, len(pkl_files)))
    try:
        with open(pkl_file, "rb") as f:
            data = pkl.load(f)
    except BaseException:
        print("loading pkl-file '{}' failed.".format(pkl_file))
        continue

    # generel information
    sample_step = data["sample_id"]
    dataset_name = data["data"]

    # data
    img = data["sample"]
    x_shape = data["dim"]
    label_arr = data["labels"]
    num_annotation = data["annotations"]
    num_classes = data["num_classes"]

    # model setup
    rank = data["rank"]

    # prediction
    mean = np.reshape(data["mean"], [-1])
    cov_diag = np.reshape(data["cov_diag"], [-1])
    cov_factor = np.reshape(data["cov_factor"], [-1, rank])

    [H, W] = x_shape[-2:]

    factor_model = FactorModel(
        flat_mean=mean,
        flat_loadings=cov_factor,
        flat_diag=cov_diag,
        data_shape=x_shape,
        num_classes=num_classes,
    )
    data["rot_mats"] = {}
    for rep_no in range(num_repetitions):

        for j, rotation in enumerate(rotations):

            changed = True

            print(
                "    {}/{}  -  {}   -   Repetition {}".format(
                    j + 1, len(rotations), rotation.name, rep_no
                )
            )

            try:
                improved = factor_model.try_improve_rotation(rotation=rotation)
                if improved:
                    print("Rotation improved!")
            except Exception as e:
                changed = False
                print("Rotation failed for ", rotation)
                print(e)
                traceback.print_exc()

            if changed:
                data["rot_mats"][rotation] = factor_model.get_rotation_matrix(
                    rotation=rotation
                )

                data["dstats"] = factor_model.dstats

    with open(pkl_file, "wb") as f:
        pkl.dump(data, f)

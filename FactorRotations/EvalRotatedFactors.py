"""
The following three scripts process a SSN prediction and are supposed 
to be run one after another: 

1. PreComputeRotations.py
2. [This Script] - EvalRotatedFactors.py
3. PrintResults.py 

This script evaluates all rotations found in loaded pickle files with respect to different
metrics as l1-norm of resulting loadings, l1-norm of resulting flow probabilities, 
sparsity, loading similarity and others. 
The results are saved within the pickle file for later aggreation with other samples
and visualization. 

Parameters:
--pkl_source   :    single pickle file or folder containing pickle files
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
from factormodel import FactorModel
from factor_metrics import *
import numpy as np
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

args = parser.parse_args()
pkl_source = args.pkl_source

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


print("Evaluate {} pkl-files.".format(len(pkl_files)))

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
    num_logits = num_classes if num_classes > 1 else 2

    # model setup
    rank = data["rank"]

    # prediction
    mean = np.squeeze(data["mean"])
    cov_diag = data["cov_diag"]
    cov_factor = data["cov_factor"]

    [H, W] = x_shape[-2:]

    factor_model = FactorModel(
        flat_mean=mean,
        flat_loadings=cov_factor,
        flat_diag=cov_diag,
        data_shape=x_shape,
        num_classes=num_classes,
    )

    factor_model.add_rotations_from_dict(data)

    mean = np.reshape(
        mean,
        (factor_model.w,
         factor_model.h,
         factor_model.num_logits))
    mean_pred = np.argmax(mean, -1).astype(np.int64)

    ##### Start Evaluation #####
    result_dict = {}

    num_samples = 100
    label_id = np.random.randint(low=0, high=num_annotation)

    z = np.random.normal(loc=0, scale=1, size=(factor_model.rank, num_samples))
    eps = np.random.normal(
        loc=0,
        scale=1,
        size=(
            num_samples,
            factor_model.w *
            factor_model.h *
            factor_model.num_logits),
    )

    samples_with_diag = factor_model.get_logits_from_sample(
        num_samples=num_samples, z=z, eps=eps, use_diag=True
    )
    samples_without_diag = factor_model.get_logits_from_sample(
        num_samples=num_samples, z=z, eps=eps, use_diag=False
    )

    stoch_pred_with_diag = np.argmax(
        samples_with_diag,
        axis=-
        1).astype(
        np.int64)
    stoch_pred_without_diag = np.argmax(
        samples_without_diag,
        axis=-
        1).astype(
        np.int64)

    # Compute metrics for diagonal evaluation
    l1_flowprobs_without_diag, l1_flowprobs_with_diag = factor_model.get_fps_full(
        plot=False, m=num_samples)

    l1_flow_probs_diff = np.linalg.norm(
        l1_flowprobs_with_diag - l1_flowprobs_without_diag, ord=1
    )
    l1_flowprobs_with_diag = np.linalg.norm(l1_flowprobs_with_diag, ord=1)
    l1_flowprobs_without_diag = np.linalg.norm(
        l1_flowprobs_without_diag, ord=1)

    result_dict["l1_flowprobs_without_diag"] = l1_flowprobs_without_diag
    result_dict["l1_flowprobs_with_diag"] = l1_flowprobs_with_diag
    result_dict["l1_flow_probs_diff"] = l1_flow_probs_diff

    # Compute Sample Diversity
    result_dict["sample_diversity"] = sample_diversity(
        stoch_pred_with_diag, num_classes=num_classes
    )
    result_dict["sample_diversity_without_diag"] = sample_diversity(
        stoch_pred_without_diag, num_classes=num_classes
    )

    # Evaluate Factors
    result_dict["hoyer_weighted"] = {}
    result_dict["loadingwise_L1"] = {}
    result_dict["loadingwise_SD"] = {}
    result_dict["pairwise_cosine_similarity"] = {}
    result_dict["pairwise_seperation_L1"] = {}

    for rotation in factor_model.rot_mats:
        factor_model.rotate(rotation)
        result_dict["hoyer_weighted"][rotation] = hoyer_weighted(factor_model)
        result_dict["loadingwise_L1"][rotation] = loadingwise_L1(factor_model)
        result_dict["loadingwise_SD"][rotation] = loadingwise_SD(factor_model)
        result_dict["pairwise_cosine_similarity"][
            rotation
        ] = pairwise_cosine_similarity(factor_model)
        result_dict["pairwise_seperation_L1"][rotation] = pairwise_seperation_L1(
            factor_model)

    print("{}/{}  -   {}".format(i + 1, len(pkl_files), pkl_file))
    print(result_dict)
    data["evaluation"] = result_dict
    with open(pkl_file, "wb") as f:
        pkl.dump(data, f)

"""
The following three scripts process a SSN prediction and are supposed 
to be run one after another: 

1. PreComputeRotations.py
2. EvalRotatedFactors.py
3. [This Script] - PrintResults.py 

This file load all pickle files from a given folder and aggregates
the performance of the single pickle samples (each stored in one pickle file).
Based on this aggregated data different performance plots as seen in the paper
are created and saved. In a folder "plots" next to the pickle source folder. 

Parameters:
--pkl_source   :    folder containing pickle files"
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
from factormodel import Rotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import pickle as pkl
import argparse

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# define and parse arguments
parser = argparse.ArgumentParser()

# config
parser.add_argument(
    "--pkl_source",
    type=str,
    help="folder containing pickle files",
)

args = parser.parse_args()
pkl_source = args.pkl_source

pkl_files = sorted([os.path.join(pkl_source, f)
                    for f in os.listdir(pkl_source) if f.endswith(".pkl")])

print(f"{len(pkl_files)} Pickle Files found!")

res_folder = os.path.join(Path(pkl_source).parent, "plots")
os.makedirs(res_folder, exist_ok=True)

# handling colors, orders and names
color_list = {
    "Unrotated": "black",
    "PCA": "purple",
    "Varimax": "green",
    "FP-Varimax": "green",
    "Equamax": "red",
    "FP-Equamax": "red",
    "Quartimax": "blue",
    "FP-Quartimax": "blue",
}

order_list = {
    "Unrotated": 0,
    "PCA": 1,
    "Varimax": 2,
    "FP-Varimax": 5,
    "Equamax": 3,
    "FP-Equamax": 6,
    "Quartimax": 4,
    "FP-Quartimax": 7,
}

label_names = {
    "ORIGINAL": "Unrotated",
    "PCA": "PCA",
    "VARIMAX": "Varimax",
    "FPVARIMAX": "FP-Varimax",
    "EQUAMAX": "Equamax",
    "FPEQUAMAX": "FP-Equamax",
    "QUARTIMAX": "Quartimax",
    "FPQUARTIMAX": "FP-Quartimax",
    "Unrotated": "Unrotated",
    "Varimax": "Varimax",
    "FP-Varimax": "FP-Varimax",
    "Equamax": "Equamax",
    "FP-Equamax": "FP-Equamax",
    "Quartimax": "Quartimax",
    "FP-Quartimax": "FP-Quartimax",
}


basic_metrics = {}
df_basic_metrics = {}
rotation_metrics = {}


file_count = 0


missed = 0
s_list = []
for i, pkl_file in enumerate(pkl_files):
    file_count += 1

    print(f"File {i+1}/{len(pkl_files)}")
    try:
        with open(pkl_file, "rb") as f:
            data = pkl.load(f)["evaluation"]
    except BaseException:
        print("loading pkl '{}' failed.".format(pkl_file))
        missed += 1
        continue
    if i == 0:
        for k in data:
            if isinstance(data[k], dict):
                for r in data[k]:
                    if r not in rotation_metrics:
                        rotation_metrics[r] = {}
                    rotation_metrics[r][k] = []
            else:
                basic_metrics[k] = []

    if ("l1_flowprobs_without_diag" not in data) or (
        "l1_flowprobs_with_diag" not in data
    ):
        print("diagonal evaluation is missing in '{}'.".format(pkl_file))
        missed += 1
        continue

    if "loadingwise_L1" not in data:
        print("loadingwise_L1 is missing in '{}'.".format(pkl_file))
        missed += 1
        continue

    con = False
    print(data)
    print(data["loadingwise_L1"])
    for k in data["loadingwise_L1"]:
        print(k)
        if len(data["loadingwise_L1"][k]) == 0:
            con = True
    if con:
        print("Not all rotations given for loadingwise_L1 in '{}'.".format(pkl_file))
        missed += 1
        continue

    for k in data:
        if isinstance(data[k], dict):

            for r in data[k]:
                if isinstance(data[k][r], list) and len(data[k][r]) == 0:
                    continue
                elif k not in rotation_metrics[r]:
                    rotation_metrics[r][k] = [data[k][r]]
                else:
                    rotation_metrics[r][k].append(data[k][r])
        else:
            basic_metrics[k].append(data[k])

print("Overview")
print("###### {} Files found! {} missed to load. ######".format(file_count, missed))
print(
    "    {}: {}+-{} (over {} samples)".format(
        "l1_flowprobs_without_diag",
        np.mean(basic_metrics["l1_flowprobs_without_diag"]),
        np.std(basic_metrics["l1_flowprobs_without_diag"]),
        len(basic_metrics["l1_flowprobs_without_diag"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "l1_flowprobs_with_diag",
        np.mean(basic_metrics["l1_flowprobs_with_diag"]),
        np.std(basic_metrics["l1_flowprobs_with_diag"]),
        len(basic_metrics["l1_flowprobs_with_diag"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "l1_relative",
        np.mean(
            np.array(basic_metrics["l1_flowprobs_without_diag"])
            / (np.array(basic_metrics["l1_flowprobs_with_diag"]) + 10e-20)
        ),
        np.std(
            np.array(basic_metrics["l1_flowprobs_without_diag"])
            / (np.array(basic_metrics["l1_flowprobs_with_diag"]) + 10e-20)
        ),
        len(basic_metrics["l1_flowprobs_with_diag"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "l1_diff",
        np.mean(
            np.abs(
                np.array(basic_metrics["l1_flowprobs_with_diag"])
                - np.array(basic_metrics["l1_flowprobs_without_diag"])
            )
        ),
        np.std(
            np.abs(
                np.array(basic_metrics["l1_flowprobs_with_diag"])
                - np.array(basic_metrics["l1_flowprobs_without_diag"])
            )
        ),
        len(basic_metrics["l1_flowprobs_with_diag"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "l1_diff_relative",
        np.mean(
            np.abs(
                np.array(basic_metrics["l1_flowprobs_with_diag"])
                - np.array(basic_metrics["l1_flowprobs_without_diag"])
            )
            / (np.array(basic_metrics["l1_flowprobs_with_diag"]) + 10e-20)
        ),
        np.std(
            np.abs(
                np.array(basic_metrics["l1_flowprobs_with_diag"])
                - np.array(basic_metrics["l1_flowprobs_without_diag"])
            )
            / (np.array(basic_metrics["l1_flowprobs_with_diag"]) + 10e-20)
        ),
        len(basic_metrics["l1_flowprobs_with_diag"]),
    )
)

print("    ############# SAMPLE DIVERSITY #############")
print(
    "    {}: {}+-{} (over {} samples)".format(
        "sample_diversity_with_diag",
        np.mean(basic_metrics["sample_diversity"]),
        np.std(basic_metrics["sample_diversity"]),
        len(basic_metrics["sample_diversity"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "sample_diversity_without_diag",
        np.mean(basic_metrics["sample_diversity_without_diag"]),
        np.std(basic_metrics["sample_diversity_without_diag"]),
        len(basic_metrics["sample_diversity_without_diag"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "sample_diversity_relative",
        np.mean(
            np.array(basic_metrics["sample_diversity_without_diag"])
            / (np.array(basic_metrics["sample_diversity"]) + 10e-20)
        ),
        np.std(
            np.array(basic_metrics["sample_diversity_without_diag"])
            / (np.array(basic_metrics["sample_diversity"]) + 10e-20)
        ),
        len(basic_metrics["sample_diversity"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "sample_diversity_diff",
        np.mean(
            np.abs(
                np.array(basic_metrics["sample_diversity"])
                - np.array(basic_metrics["sample_diversity_without_diag"])
            )
        ),
        np.std(
            np.abs(
                np.array(basic_metrics["sample_diversity"])
                - np.array(basic_metrics["sample_diversity_without_diag"])
            )
        ),
        len(basic_metrics["sample_diversity"]),
    )
)

print(
    "    {}: {}+-{} (over {} samples)".format(
        "sample_diversity_diff_relative",
        np.mean(
            np.abs(
                np.array(basic_metrics["sample_diversity"])
                - np.array(basic_metrics["sample_diversity_without_diag"])
            )
            / (np.array(basic_metrics["sample_diversity"]) + 10e-20)
        ),
        np.std(
            np.abs(
                np.array(basic_metrics["sample_diversity"])
                - np.array(basic_metrics["sample_diversity_without_diag"])
            )
            / (np.array(basic_metrics["sample_diversity"]) + 10e-20)
        ),
        len(basic_metrics["sample_diversity"]),
    )
)

print(" ####################### ")

r_names = [label_names[r.name] for r in rotation_metrics]


def print_metric_vs_threshold(
        key,
        b_metrics,
        r_metrics,
        norm_type="",
        tau_max=1.0):

    curves = {}

    for r in r_metrics:

        ordered_value = [np.sort(r_metrics[r][key][i])[::-1]
                         for i in range(len(r_metrics[r][key]))]

        # assert that loadings are sorted correctly
        assert np.all(np.diff(ordered_value, axis=-1) <= 0)

        if key == "loadingwise_SD":
            norm_vals = np.expand_dims(b_metrics["sample_diversity"], -1)
        else:
            if norm_type == "fp_without_diag":
                norm_vals = np.expand_dims(
                    b_metrics["l1_flowprobs_without_diag"], axis=-1
                )
            else:
                norm_vals = np.linalg.norm(
                    r_metrics[r][key], axis=-1, keepdims=True, ord=1
                )

        mean_curve = np.array(
            list(
                map(
                    lambda rho: np.sum(
                        rho *
                        norm_vals <= ordered_value,
                        axis=-
                        1),
                    np.arange(
                        start=0,
                        stop=tau_max +
                        0.01,
                        step=0.01),
                )))

        curves[label_names[r.name]] = mean_curve

    return curves


def pairwise_1d_range(
        key,
        key_l1,
        b_metrics,
        r_metrics,
        tau_range=0.0,
        norm_type=""):
    def get_rho_list(idx_list, rho, depth=0, best_found=0):

        if depth + len(idx_list) <= best_found:
            return best_found

        for idx in idx_list:
            best_found = get_rho_list(
                [i for i in idx_list if ((i > idx) and (cos_sim_mat[i, idx] <= rho))],
                rho=rho,
                depth=depth + 1,
                best_found=np.max([best_found, depth + 1]),
            )

        return best_found

    aoc_dict = {}

    for r in r_metrics:

        aoc_dict[label_names[r.name]] = []

        # get absolute cosine similarities
        pairwise_abs_cosine_similarity = np.abs(r_metrics[r][key])

        # get loading-wise L1-norm
        loadingwise_l1 = r_metrics[r][key_l1]

        if norm_type == "fp_without_diag":
            norm_vals = b_metrics["l1_flowprobs_without_diag"]
        elif norm_type == "fp_with_diag":
            norm_vals = b_metrics["l1_flowprobs_with_diag"]
        else:
            norm_vals = np.sum(loadingwise_l1, axis=-1)

        counter = 1
        # iterate over samples
        for sub_a, sub_b, norm_val in zip(
            pairwise_abs_cosine_similarity, loadingwise_l1, norm_vals
        ):
            # print(counter, "/", len(pairwise_abs_cosine_similarity))
            counter += 1

            # fill table with cosine similarities
            cos_sim_mat = np.eye(10)
            sample_ids = np.add(np.triu_indices(10, 1), 0).T
            for i, idx in enumerate(sample_ids):
                cos_sim_mat[idx[0], idx[1]] = sub_a[i]
                cos_sim_mat[idx[1], idx[0]] = sub_a[i]

            # make L1 contritbuion relative
            sample_l1 = sub_b / np.max([norm_val, 10e-20])

            rho_list_unique = np.sort(np.unique(np.reshape(cos_sim_mat, -1)))
            tau_pointer = len(sample_l1) - 1
            res_values = []

            mean_curve = []
            mean_curve_norm = []

            tau_pointer_changed = True
            for tau in tau_range:

                # first time active change, after words only if realy changed
                tau_pointer_changed = tau == 0

                while tau_pointer > 0 and sample_l1[tau_pointer] < tau:
                    # res_values.append(res_values[-1])
                    tau_pointer -= 1
                    tau_pointer_changed = True

                if not tau_pointer_changed:

                    if sample_l1[tau_pointer] < tau:
                        res_values.append(np.zeros_like(res_values[-1]))
                    else:
                        res_values.append(res_values[-1])
                    continue

                idx_list = np.arange(start=0, stop=tau_pointer + 1)
                # increase tau pointer for next iteration
                res_value = np.zeros((1001,))

                rho_list_pointer = 0

                for j, rho in enumerate(np.arange(0, 1.001, 0.001)):
                    sub_update = False
                    while (
                        rho_list_pointer < len(rho_list_unique)
                        and rho_list_unique[rho_list_pointer] <= rho
                    ):
                        sub_update = True
                        rho_list_pointer += 1

                    if j == 0:
                        sub_update = True

                    if sub_update:
                        if j > 0:
                            res_value[j] = get_rho_list(
                                idx_list, rho=rho, best_found=res_value[j - 1]
                            )
                        else:
                            res_value[j] = get_rho_list(idx_list, rho=rho)
                    elif j > 0:
                        res_value[j] = res_value[j - 1]

                mean_curve.append(np.copy(res_value))
                mean_curve_norm.append(
                    np.copy(res_value) / (len(idx_list) + 10e-20))

                res_values.append(np.mean(mean_curve_norm[-1]))

            aoc_dict[label_names[r.name]].append(res_values)

    return [aoc_dict]


x = np.arange(0.000, 2.5, 0.005)
x_sd = np.arange(0.0, 1.5, 0.005)

# load or compute AUC list
auroc_list = pairwise_1d_range(
    key="pairwise_cosine_similarity",
    key_l1="loadingwise_L1",
    b_metrics=basic_metrics,
    r_metrics=rotation_metrics,
    tau_range=x,
    norm_type="fp_without_diag",
)

# compute l1 curves
l1_list = print_metric_vs_threshold(
    key="loadingwise_L1",
    b_metrics=basic_metrics,
    r_metrics=rotation_metrics,
    norm_type="fp_without_diag",
    tau_max=2.5,
)

l1_means = {}
l1_stds = {}
sd_means = {}
sd_stds = {}
auroc_means = {}
auroc_stds = {}


for r in auroc_list[0]:
    auroc_means[label_names[r]] = np.mean(auroc_list[0][r], axis=0)
    auroc_stds[label_names[r]] = np.std(auroc_list[0][r], axis=0)
    l1_means[label_names[r]] = np.mean(l1_list[r], axis=-1)
    l1_stds[label_names[r]] = np.std(l1_list[r], axis=-1)

label_size = 22
tick_size = 20
legend_size = 22
title_size = 24

# L1 Relevance with std
n_cols = 4
n_rows = 2
fig, axs = plt.subplots(
    nrows=n_rows, ncols=n_cols, figsize=(
        7 * n_cols, 5 * n_rows))
counter = 1

for r in auroc_list[0]:
    column = counter // 2
    row = counter % 2

    name = label_names[r]
    sign = "-" if name.startswith("FP") else "--"

    name = label_names[r]

    if n_rows == 1:
        ax_sub = axs[column]
    else:
        ax_sub = axs[row, column]

    axs[0, 0].plot(
        np.arange(0.000, 2.51, 0.01),
        l1_means[name],
        sign,
        label=name,
        color=color_list[name],
    )
    axs[0, 0].set_xlabel(r"$\tau$", fontsize=label_size)

    ax_sub.plot(
        np.arange(0.000, 2.51, 0.01),
        l1_means[name],
        sign,
        label=name,
        color=color_list[name],
    )
    ax_sub.fill_between(
        np.arange(0.000, 2.51, 0.01),
        l1_means[r] + l1_stds[r],
        l1_means[r] - l1_stds[r],
        facecolor=color_list[name],
        alpha=0.5,
    )

    ax_sub.set_xlabel(r"$\tau$", fontsize=label_size)
    axs[row, 0].set_ylabel(r"$n_\tau$", fontsize=label_size)

    axs[0, 0].tick_params(axis="both", labelsize=tick_size)
    ax_sub.tick_params(axis="both", labelsize=tick_size)

    counter += 1

handles_s, labels_s = axs[0, 0].get_legend_handles_labels()
labels_s, handles_s = zip(
    *sorted(zip(labels_s, handles_s), key=lambda t: order_list[t[0]])
)

axs[1, 2].legend(
    handles_s,
    labels_s,
    fontsize=legend_size,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    fancybox=False,
    shadow=False,
    ncol=7,
)

plt.tight_layout()

plt.savefig(os.path.join(res_folder, "Relevance_l1.png"))
plt.savefig(os.path.join(res_folder, "Relevance_l1.pdf"))
plt.close()


# AUC with Std
n_cols = 4
n_rows = 2
fig, axs = plt.subplots(
    nrows=n_rows, ncols=n_cols, figsize=(
        7 * n_cols, 5 * n_rows))
counter = 1

for r in auroc_list[0]:
    column = counter // 2
    row = counter % 2

    name = label_names[r]
    sign = "-" if name.startswith("FP") else "--"

    if n_rows == 1:
        ax_sub = axs[column]
    else:
        ax_sub = axs[row, column]

    axs[0, 0].plot(x, auroc_means[r], sign, label=name, color=color_list[name])
    axs[0, 0].set_xlabel(r"$\tau$", fontsize=label_size)

    ax_sub.plot(x, auroc_means[r], sign, label=name, color=color_list[name])
    ax_sub.fill_between(
        x,
        auroc_means[r] + auroc_stds[r],
        auroc_means[r] - auroc_stds[r],
        facecolor=color_list[name],
        alpha=0.5,
    )

    ax_sub.set_xlabel(r"$\tau$", fontsize=label_size)
    axs[row, 0].set_ylabel(r"$n_\tau$", fontsize=label_size)

    ax_sub.tick_params(axis="both", labelsize=tick_size)
    axs[0, 0].tick_params(axis="both", labelsize=tick_size)
    axs[0, 0].set_ylim(0, 1)

    ax_sub.set_ylim(0, 1)
    counter += 1


handles_s, labels_s = axs[0, 0].get_legend_handles_labels()
labels_s, handles_s = zip(
    *sorted(zip(labels_s, handles_s), key=lambda t: order_list[t[0]])
)

axs[0, 1].legend(
    handles_s,
    labels_s,
    fontsize=22,
    loc="upper center",
    bbox_to_anchor=(1.1, 1.3),
    ncol=7,
)

fig.tight_layout()
plt.savefig(os.path.join(res_folder, "AUC_cosine_similarity.png"))
plt.savefig(os.path.join(res_folder, "AUC_cosine_similarity.pdf"))
plt.close()


# Print Violine Plots for Sparsity
print_list = ["hoyer_weighted"]
name_list = ["Weighted Hoyer Measure"]
idx = np.arange(len(rotation_metrics))
idx = sorted(idx, key=lambda x: -
             order_list[label_names[list(rotation_metrics.keys())[x].name]])
rotations_sorted = sorted(list(rotation_metrics.keys()),
                          key=lambda x: -order_list[label_names[x.name]])

for p, name in zip(print_list, name_list):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    df_a = []
    for k in rotations_sorted:
        df_a.append(rotation_metrics[k][p])

    color = [color_list[label_names[k.name]] for k in rotations_sorted]
    v_plt = axs.violinplot(dataset=df_a, vert=False, data=None, showmeans=True)

    for i, o in enumerate(rotations_sorted):
        v_plt["bodies"][i].set_facecolor(color_list[label_names[o.name]])
        v_plt["cmeans"].set_edgecolor("gray")

    axs.set_yticks(np.arange(1, len(df_a) + 1))
    axs.set_xlim(-0.01, 1.01)
    axs.set_xlabel(name)
    axs.set_xticks(np.arange(0, 1.1, 0.2))
    axs.set_yticklabels([label_names[r.name] for r in rotations_sorted])

    plt.tight_layout()
    print("save ", f"Overview_{p}.png")
    plt.savefig(os.path.join(res_folder, f"Overview_{p}.png"))
    plt.savefig(os.path.join(res_folder, f"Overview_{p}.pdf"))
    plt.close()


print("Result plots saved into '{}'.".format(res_folder))

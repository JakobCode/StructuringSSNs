import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.realpath("dummy.ipynb")).parent.parent))
sys.path.append(os.path.join(str(Path(os.path.realpath("dummy.ipynb")).parent.parent), "FactorRotations", "utils"))

from scipy.stats import norm
from factormodel import FactorModel, Rotations
from matplotlib import colors
import numpy as np
import pickle as pkl

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
def mycmap(dataset):
    if dataset == "CamVid":
            color_list = [
                "#808080",
                "#800000",
                "#C0C080",
                "#804080",
                "#3C28DE",
                "#808000",
                "#C08080",
                "#404080",
                "#400080",
                "#404000",
                "#0080C0",
                "#000000",
            ]
            cmap = colors.ListedColormap(color_list)
            neutral_color = (1, 1, 1)

    elif dataset == "SEN12MS":
        color_list = [
            "#009900",
            "#c6b044",
            "#fbff13",
            "#b6ff05",
            "#27ff87",
            "#c24f44",
            "#a5a5a5",
            "#69fff8",
            "#f9ffa4",
            "#1c0dff",
            "#ffffff",
        ]
        cmap = colors.ListedColormap(color_list)

    else:
        color_list = ["#0000ff", "#ff0000"]
        cmap = colors.ListedColormap(color_list)
        neutral_color = (1, 1, 1)

    return cmap

class DataHandler:
    def __init__(self, sample_path):
        self.cmap = None
        self.factor_model = None

        self.rot_translate = None

        self.rot_translate_rev = None

        self.available_rotations = None

        self.cov_temperature = None
        self.factor_weights = None
        self.diag_weights = None
        self.fill_dh(sample_path)
    

    def fill_dh(self, sample_path):
        with open(sample_path, "rb") as f:
            data = pkl.load(f) 

        self.data_set = data["data"]
        labels = data["labels"]
        flat_mean = np.reshape(data["mean"], [-1])
        flat_loadings = data["cov_factor"]
        flat_diag = np.reshape(data["cov_diag"], [-1])
        data_shape = data["dim"]
        num_classes = data["num_classes"]
        sample = data["sample"]
        
        if np.shape(labels)[-1] == 4: 
            labels = np.transpose(labels, [2,0,1])

        if self.data_set == "SEN12MS": 
            sample = np.transpose(sample[[2,1,0],:,:], axes=(1,2,0))
            sample = np.clip(sample * 10000, 0, 2000) / 2000
        if self.data_set == "CamVid": #
            sample = np.transpose(sample, axes=(1,2,0))
            std = np.array([[[0.27413549931506, 0.28506257482912, 0.28284674400252]]]) 
            m = np.array([[[0.41189489566336, 0.4251328133025, 0.4326707089857]]])
            sample = sample * std + m
            labels = np.expand_dims(labels,0)

        self.factor_model = FactorModel( 
                    flat_mean=flat_mean, 
                    flat_loadings=flat_loadings, 
                    flat_diag=flat_diag, 
                    data_shape=data_shape, 
                    num_classes=num_classes, 
                    num_logits=None,
                    sid=-1, sample=sample, 
                    labels=labels,
                    pkl_path=None, 
                    cmap=mycmap(self.data_set))

        self.cmap=mycmap(self.data_set)
        self.factor_model.add_rotations_from_dict(data)

        self.rot_translate = {Rotations.ORIGINAL: "Unrotated", Rotations.VARIMAX: "Varimax", Rotations.EQUAMAX: "Equamax", Rotations.QUARTIMAX: "Quartimax", 
                              Rotations.FPVARIMAX: "FP-Varimax", Rotations.FPEQUAMAX: "FP-Equamax", Rotations.FPQUARTIMAX: "FP-Quartimax"}

        self.rot_translate_rev = {"Unrotated": Rotations.ORIGINAL, "Varimax": Rotations.VARIMAX, "Equamax": Rotations.EQUAMAX, "Quartimax": Rotations.QUARTIMAX,
                                  "FP-Varimax": Rotations.FPVARIMAX, "FP-Equamax": Rotations.FPEQUAMAX, "FP-Quartimax": Rotations.FPQUARTIMAX}

        self.available_rotations = list(self.rot_translate_rev.keys())# = [self.rot_translate[r] for r in self.factor_model.rot_mats.keys() if r.name != "PCA"]

        #for rot in self.rot_translate_rev:
        #    if self.rot_translate_rev[rot] not in self.factor_model.rot_mats:
        #        del self.available_rotations[rot]

        # Scalling Factors
        self.cov_temperature = 1.0
        self.factor_weights = np.zeros(self.factor_model.rank) 
        self.diag_weights = np.zeros(self.factor_model.w*self.factor_model.h * self.factor_model.num_logits) 

    def update_example(self, pkl_path):
        self.fill_dh(pkl_path)

    def get_sample(self):
        a = np.squeeze(self.factor_model.get_logits_from_sample(z=self.factor_weights, eps=self.diag_weights))
        return a

    def get_mean_pred(self):
        a = np.squeeze(self.factor_model.mean)
        a = np.reshape(a, [self.factor_model.w, self.factor_model.h, self.factor_model.num_logits])
        return np.argmax(a,-1)

    def get_prediction(self):
        return np.argmax(self.get_sample(),-1)

    def scale_covariance(self, new_value):
        self.cov_temperature = new_value

    def scale_factor(self, factor_id, new_value):
        self.factor_weights[factor_id-1] = new_value

    def reset(self):
        self.factor_weights = np.zeros(self.factor_model.rank)
        self.diag_weights = np.zeros(self.factor_model.w * self.factor_model.h * self.factor_model.num_logits) 
        self.cov_temperature = 1.0

    def resample(self):
        self.factor_weights = np.random.normal(loc=0, scale=1, size=self.factor_model.rank)
        self.diag_weights = np.random.normal(loc=0, scale=1, size=self.factor_model.w * self.factor_model.h * self.factor_model.num_logits)

    def rotate_factors(self, new_rotation):
        self.factor_model.rotate(self.rot_translate_rev[new_rotation])

    def plot_factors(self):
        if self.data_set == "CamVid":
            upscale = 4
        else: 
            upscale = 2

        factors_pos, factors_neg = self.factor_model.get_onesidedflowprobs(rotations=[self.factor_model.cur_rotation], 
                                                                           upscale=upscale)

        return np.clip(factors_pos,0,1), np.clip(factors_neg,0,1)

    def plot_sample_and_labels(self, axs):
        if self.factor_model.sample is None:
            print("No sample available")
        else:
            sample = self.factor_model.sample
            axs[0].imshow(sample)

        if self.factor_model.labels is None:
            print("No labels available")
        else:
            for i in range(1, len(axs)):
                axs[i].clear()
                axs[i].axis('off')
                axs[i].set_xticklabels([])
                axs[i].set_yticklabels([])
                axs[i].set_aspect('equal')

            for i in range(len(self.factor_model.labels)):
                axs[i+1].imshow(self.cmap(self.factor_model.labels[i].astype(int)))

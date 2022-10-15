"""


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
from FactorRotations.utils.rotator import Rotator, get_classprobs, get_flowprobs
from scipy.stats import ortho_group
from scipy.special import softmax
from matplotlib import cm, colors
from einops import rearrange
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from enum import Enum, auto
import pickle
from tkinter import E


def cast_rotations(rot):
    """
    Minor helper function which might be needed when signature of loaded
    rotations does not fit the signiture of the one defined in this script. 
    
    Parameters
    ----------
    rot   :    Rotation     Any file that contains the attribute name
    """

    for r in Rotations:
        if r.name == rot.name:
            return r

    raise Exception("No rotaiton found")


def reduce_bgclass(mean, loading_mat, shape):
    """
    Helper function that remove the background class for binary classification 
    problems.

    Parameters:
    -----------
    mean          :   nparray     Mean of the factor model
    loading_mat   :   nparray     Loading matrix of factor model
    shape         :   int[]       Shape of the underlying (non-flattened) mean
    """

    w, h, num_c = shape
    loading_mat = rearrange(loading_mat, "(p c) k -> p c k", c=num_c)
    fcs_reduced = np.empty(
        (w * h, num_c - 1, loading_mat.shape[-1]), dtype=loading_mat.dtype
    )
    for i in range(1, num_c):
        fcs_reduced[:, i - 1, :] = loading_mat[:, i, :] - loading_mat[:, 0, :]
    #    print(rearrange(loading_mat, "p c k -> (p c) k"))
    #    print(rearrange(fcs_reduced, "p c k -> (p c) k"))
    mean = rearrange(mean, "(p c) -> p c", c=num_c)
    tmp = mean[:, 0].repeat(num_c - 1, 1).T
    mean_reduced = mean[:, 1:] - tmp

    return mean_reduced, fcs_reduced


def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A.
    
    Parameters:
    -----------
    A   :   nparray     Matrix to be orthogonalized."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A



def get_sortidx_bynorm(loading_mat, ord=1):
    """
    Sort the columns of a matrix by the column-wise l1-norm
    and return the sorted indices. 

    Parameters:
    ----------
    loading_mat   :   nparray      Matrix to be processed
    ord           :   int          Apply l_ord norm on columns (default: 1)
    """

    k = loading_mat.shape[1]
    norms = [np.linalg.norm(loading_mat[:, i], ord) for i in range(k)]
    idx_sorted = sorted(list(range(k)), key=lambda i: norms[i], reverse=True)

    return idx_sorted


def get_color_map(dataset):
    """
    Get color map of different data sets. This colormap is needed
    for the visualization methods included in the class FactorModel.
    If not known dataset is given, a binary colormap is returned. 

    Parameters: 
    -----------
    dataset  :  str     Which dataset ["CamVid", "SEN12MS"]
    """

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
        neutral_color = (0, 0, 0)

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
        neutral_color = (0, 0, 0)

    else:
        color_list = ["#0000ff", "#ff0000"]
        cmap = colors.ListedColormap(color_list)
        neutral_color = (0, 0, 0)

    return cmap, neutral_color


"""
Dictionary for ordering the different rotations.
"""
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

"""
Dictionary for mapping rotation names to labels in visualizations. 
"""
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


class Rotations(Enum):
    """
    Enums for different types of rotations (not all implemented yet)
    """
    ORIGINAL = auto()       # implemented
    VARIMAX = auto()        # implemented
    DVARIMAX = auto()       # not implemented
    EQUAMAX = auto()        # implemented
    PROJECTION = auto()     # not implemented
    PCA = auto()            # not implemented
    FPVARIMAX = auto()      # implemented
    FPEQUAMAX = auto()      # implemented
    QUARTIMAX = auto()      # implemented
    FPQUARTIMAX = auto()    # implemented
    FP1VARIMAX = auto()     # not implemented
    FP1EQUAMAX = auto()     # not implemented
    FP1QUARTIMAX = auto()   # not implemented


"""
Dictionary for mapping from rotation name to rotation. 
"""
rot_mapping = {
    "ORIGINAL": Rotations.ORIGINAL,
    "PCA": Rotations.PCA,
    "VARIMAX": Rotations.VARIMAX,
    "FPVARIMAX": Rotations.FPVARIMAX,
    "EQUAMAX": Rotations.EQUAMAX,
    "FPEQUAMAX": Rotations.FPEQUAMAX,
    "QUARTIMAX": Rotations.QUARTIMAX,
    "FPQUARTIMAX": Rotations.FPQUARTIMAX,
}


class FactorModel:
    """
    Class for parameters of a factor model
        N(mean, FactorLoadings*FactorLoadings.T + Diag)

    provides interface for various rotations of the factors
    """

    def __init__(
        self,
        flat_mean,
        flat_loadings,
        flat_diag,
        data_shape,
        num_classes,
        num_logits=None,
        sid=-1,
        sample=None,
        labels=None,
        pkl_path=None,
        cmap=None,
        neutral_color=(1, 1, 1),
        dstats=None,
        ):
        """
        Initialization of a FactorModel instance.

        Parameters:
        -----------
        flat_mean      :  nparray                1D array of mean
        flat_loadings  :  nparray                1D array of all loadings
        flat_diag      :  nparray                1D array of diagonal
        data_shape     :  list<int>              shape of output data
        num_classes    :  int                    number of classes in seg. problem
        num_logits     :  int, optional          number of logtis (only for binary - 1 or 2)
        sid            :  int, optional          sample id, default: -1
        sample         :  nparray, optional      input image that led to the prediction
        labels         :  nparray, optional      groundtruth or expert label(s)
        pkl_path       :  str, optional          path to pkl-file where these parameters are loaded from     
        cmap           :  colormap, optional     colormap for visualizations, default: None
        neutral_color  :  tuple<int>, optional   neutral color for background of flow probabilities
        dstats         :  dict, optional         dictionary with computation statistics from earlier runs.
        
        """
        # i split logits classes --> binary case can be single or two logits
        assert (
            (num_classes == num_logits)
            or (num_logits is None)
            or (num_logits == 2 and num_classes == 1)
        )


        # input data and labels
        self.sample = sample
        self.labels = labels
        self.pkl_path = pkl_path

        # Dimension properties
        self.num_classes = num_classes
        self.num_logits = num_logits if num_logits is not None else num_classes
        self.w = data_shape[-2]
        self.h = data_shape[-1]
        self.rank = flat_loadings.shape[-1]

        self.flatshape = flat_loadings.shape[-2]
        assert self.w * self.h * self.num_logits == self.flatshape

        # Distribution Parameters
        self.mean = flat_mean
        self.diag = flat_diag
        self.sqdiag = np.sqrt(flat_diag)

        self.loadings = {Rotations.ORIGINAL: flat_loadings}
        self.rot_mats = {Rotations.ORIGINAL: np.eye(N=self.rank, M=self.rank)}
        self.fps = {}
        self.dstats = {
            "best_runs": {},
            "sub_data": []} if dstats is None else dstats

        self.cur_rotation = Rotations.ORIGINAL

        # color map

        if cmap is None:
            self.cmap = None

        elif isinstance(cmap, colors.ListedColormap):
            self.cmap, self.neutralcolor = cmap, neutral_color
        elif isinstance(cmap, str):
            self.cmap, self.neutralcolor = get_color_map(cmap)
        else:
            raise Exception("color map not in valid format.")

        # Rotation functions
        self.rotation_funcs = {
            Rotations.VARIMAX: self._varimax,
            Rotations.FPVARIMAX: self._varimax,
            Rotations.EQUAMAX: self._equamax,
            Rotations.FPEQUAMAX: self._equamax,
            Rotations.FPQUARTIMAX: self._quartimax,
            Rotations.QUARTIMAX: self._quartimax,
        }

        self.sid = sid  # sample id (if applicable)

        self.rotator = Rotator(num_c=self.num_logits)

    def get_loadings(
        self, rotation: Rotations = None, sort=True, sort_by_flowprobs=False):
        """
        Returns the loadings of the factor model, either of a specific or the current rotations.
        The loadings can be sorted by loadings or flow probabilities.

        Parameters:
        -----------
        rotation            :   Rotations, optional     which rotation to use (defaul: None -> current rotation)
        sort                :   bool, optional          sort the loadings (default: True)
        sort_by_flowprobs   :   bool, optional          sort the loading by their flow probabilities (default: True)
        """

        if rotation is None or rotation == self.cur_rotation:
            loading_mat = self.loadings[self.cur_rotation]

        else:
            save_rotation = self.cur_rotation

            self.rotate(rotation)
            loading_mat = self.loadings[self.cur_rotation]

            assert save_rotation is not rotation
            self.rotate(save_rotation)

        if sort and not sort_by_flowprobs:
            self.sorted_idx = get_sortidx_bynorm(loading_mat)
        elif sort_by_flowprobs:
            self.sorted_idx = get_sortidx_bynorm(
                self.get_flow_probabilities(rotation=rotation, sort=False)
            )
        else:
            self.sorted_idx = sorted(np.arange(self.rank))

        return loading_mat[:, self.sorted_idx]

    def get_flow_probabilities(self, rotation: Rotations, sort=True):
        """
        Compute the factor-wise flow probabilities from a given rotation. If the rotation 
        is already pre-computed, it will be loaded, otherwise it will be computed. 

        Parameters: 
        rotation  :  Rotations      the rotation to be evaluated
        sort      :  bool           sort the factor-wise flow probabilities by relevance
        """

        if rotation not in self.fps:
            print("flow probs not found - start computation")
            loadings = self.get_loadings(rotation, sort=False)
            fps = get_flowprobs(
                self.mean,
                loadings,
                self.num_logits,
                usepool=False,
                gradient=False)
            self.fps[rotation] = fps["f"]

        if sort:
            self.sorted_idx = get_sortidx_bynorm(self.fps[rotation])
        else:
            self.sorted_idx = sorted(np.arange(self.rank))
        return self.fps[rotation][:, self.sorted_idx]

    def get_rotation_matrix(self, rotation: Rotations):
        """Loads the rotation matrix for a given rotation. 
        The rotation need to be pre-computed and stored within 
        the the FactorModel instance.
        
        Parameters:
        -----------
        rotation   :   Rotations      The rotation criteria for which 
                                      loaded matrix was optimized
        """
        return self.rot_mats[rotation]

    def get_logits_from_sample(
            self,
            z=None,
            eps=None,
            num_samples=1,
            mask=None,
            use_diag=True):
        """
        Get realizations of the factor model. Either by sampling or by using
        passed realizations of latent factor variables.

        Parameters:
        -----------
        z             :   nparray, optional     sampled latent variables for factor (defaul: None)
        eps           :   nparray, optional     sampled latent variables for the diagonal (default: None)
        num_samples   :   int, optinal          number of samples to be generated from factor model (default: 1)
        mask          :   nparray, optional     boolean mask to mask out individual factors
        use_diag      :   bool                  use the diagonal within the sampling (default: True)
        """

        loading_mat = self.get_loadings(
            self.cur_rotation, sort=True, sort_by_flowprobs=True
        )
        w = self.w
        h = self.h
        num_c = self.num_logits
        n = num_samples if num_samples is not None else 1

        if z is None:
            z = np.random.normal(loc=0, scale=1, size=(self.rank, n))
        if eps is None and use_diag:
            eps = np.random.normal(loc=0, scale=1, size=(n, w * h * num_c))

        if mask is not None:
            z = z * mask

        eta = np.expand_dims(self.mean.copy(), 1)

        if len(z.shape) == 1:
            z = np.reshape(z, [-1, 1])

        eta = eta + np.dot(loading_mat, z)

        eta = np.transpose(eta)

        if eps is not None and len(eps.shape) == 1:
            eps = np.expand_dims(eps, 0)

        if use_diag:
            eta += np.expand_dims(self.sqdiag, 0) * eps

        eta = eta.reshape((eta.shape[0], w, h, num_c))

        if eta.shape[0] == 1:
            return eta.reshape((w, h, num_c))
        else:
            return eta

    def add_rotations_from_dict(self, data, update=False):
        """
        Checks given dictionary for contained factor rotations and sets or updates
        stored factor representations in this model.

        data:       Dictionary containing the loading_mat in numpy format
        update:     If set, loading_mat will be overwritten if already contained (Default: False)
        """

        if "rot_mats" not in data:
            Warning("The given dictionary does not contain any rotation matrices!")
        else:
            for rotation in data["rot_mats"]:
                #                print(rotation)
                if rotation.name in rot_mapping:
                    if rot_mapping[rotation.name] not in self.rot_mats or update:
                        #                    print("added")
                        self.rot_mats[rot_mapping[rotation.name]
                                      ] = data["rot_mats"][rotation]

    def get_mean(self):
        """
        Returns the factor model's mean.
        """
        return self.mean

    def get_diag(self, sqrt=False):
        """
        Returns the factor model's diagonal
        """
        return self.sqdiag if sqrt else self.diag

    # *** rotations ***
    def _varimax(self, loading_mat, init_rot_mat=None, **kwargs):
        """
        Optimizes a given loading matrix with respect to the varimax objective.
        If not initial rotation matrix is given, a random one is generated. 
        
        Parameters:
        -----------
        loading_mat   :   nparray               loading matrix to apply rotation on 
        init_rot_mat  :   nparray, optional     initial rotation matrix (default: None)
        """
        if init_rot_mat is None:
            print("Random initialization.")
            init_rot_mat = ortho_group.rvs(dim=self.rank)
        varimax_loading_mat, rot_mat, dstats = self.rotator.solve_orthogonal(
            loading_mat, method="varimax", init=init_rot_mat, **kwargs
        )

        return varimax_loading_mat, rot_mat, dstats

    def _equamax(self, loading_mat_org, init_rot_mat=True, **kwargs):
        """
        Optimizes a given loading matrix with respect to the equamax objective.
        If not initial rotation matrix is given, a random one is generated. 
        
        Parameters:
        -----------
        loading_mat_org   :   nparray               loading matrix to apply rotation on 
        init_rot_mat      :   nparray, optional     initial rotation matrix (default: None)
        """

        if init_rot_mat is None:
            print("Random initialization.")
            init_rot_mat = ortho_group.rvs(dim=self.rank)

        rot_loading_mat, rot_mat, dstats = self.rotator.solve_orthogonal(
            loading_mat_org, method="equamax", init=init_rot_mat, **kwargs
        )
        return rot_loading_mat, rot_mat, dstats

    def _quartimax(self, loading_mat_org, init_rot_mat=None, **kwargs):
        """
        Optimizes a given loading matrix with respect to the quartimax objective.
        If not initial rotation matrix is given, a random one is generated. 
        
        Parameters:
        -----------
        loading_mat_org   :   nparray               loading matrix to apply rotation on 
        init_rot_mat      :   nparray, optional     initial rotation matrix (default: None)
        """

        if init_rot_mat is None:
            print("Random initialization.")
            init_rot_mat = ortho_group.rvs(dim=self.rank)

        rot_loading_mat, rot_mat, dstats = self.rotator.solve_orthogonal(
            loading_mat_org, method="quartimax", init=init_rot_mat, **kwargs
        )
        return rot_loading_mat, rot_mat, dstats



    def try_improve_rotation(self, rotation: Rotations, *args, **kwargs):
        """
        Optimizes a given loading matrix with respect to the given rotation criterion.
        The optimization prcedure is also run if the rotation is already computed. 
        The lowest objective will be saved. 

        Parameters:
        -----------
        rotation   :   Rotations      The rotation criterion to optimize 
        """

        init_rot_mat = None
        if rotation.name.startswith("FP"):
            self.rotator.toggle_use_fp(False, mean_logits=self.mean)
            _, init_rot_mat, _ = self.rotation_funcs[rotation](
                self.loadings[Rotations.ORIGINAL],
                init_rot_mat=init_rot_mat,
                max_iter=20,
                *args,
                **kwargs,
            )

            self.rotator.toggle_use_fp(True, mean_logits=self.mean)
        else:
            self.rotator.toggle_use_fp(False, mean_logits=self.mean)

        _, rot_mat, sub_dstats = self.rotation_funcs[rotation](
            self.loadings[Rotations.ORIGINAL],
            init_rot_mat=init_rot_mat,
            *args,
            **kwargs,
        )

        if sub_dstats is not None:
            sub_dstats["rotation_matrix"] = rot_mat

            if len(sub_dstats["obj"]) == 0:
                new_obj = 0
            else:
                new_obj = sub_dstats["obj"][-1]
                # assert np.all(np.diff(sub_dstats["obj"])<=0)
        else:
            new_obj = -1

        if (
            new_obj == -1
            or rotation not in self.dstats
            or len(self.dstats[rotation]) == 0
        ):
            old_obj = np.inf
            print("Current best obj.: -   (first run)")

        else:
            if (
                len(self.dstats[rotation][self.dstats["best_runs"][rotation]]["obj"])
                == 0
            ):
                old_obj = 0
            else:
                old_obj = self.dstats[rotation][self.dstats["best_runs"]
                                                [rotation]]["obj"][-1]

            print("Current best obj.: ", old_obj)

        print("Computed obj.:     ", new_obj)

        improved = old_obj > new_obj

        if rotation not in self.dstats:
            self.dstats[rotation] = []
        self.dstats[rotation].append(sub_dstats)

        if improved:
            self.rot_mats[rotation] = rot_mat
            self.loadings[rotation] = np.matmul(
                self.loadings[Rotations.ORIGINAL], self.rot_mats[rotation]
            )

            self.dstats["best_runs"][rotation] = len(self.dstats[rotation]) - 1

        return improved

    def rotate(
        self,
        rotation: Rotations,
        retry: bool = False,
        init_rot_mat=None,
        *args,
        **kwargs,
        ):
        """
        Rotates the current representation of the factor model. 

        Parameters: 
        -----------
        rotation      :  Rotations     which rotation matrix to use
        retry         :  boolean       recompute the rotation if it already exists (defaul: False)
        init_rot_mat  :  nparray       initial rotation matrix for optimization (default: None) 
        """
        
        name = rotation.name
        if rotation not in self.rot_mats or retry:
            if name.startswith("FP"):
                self.rotator.toggle_use_fp(True, mean_logits=self.mean)
            else:
                self.rotator.toggle_use_fp(False, mean_logits=self.mean)
            print(f"Rotation '%s' not found. Start optimization:" % name)

            assert rotation in Rotations, "Rotation %s unknown" % rotation
            _, rot_mat, dstats = self.rotation_funcs[rotation](
                self.loadings[Rotations.ORIGINAL], init_rot_mat, *args, **kwargs
            )

            # check if rotation works correctly
            # assert np.allclose(loading_mat, np.matmul(self.loadings[Rotations.ORIGINAL], rot_mat)), "check failed"

            self.rot_mats[rotation] = rot_mat
            self.dstats[rotation] = [dstats]
        else:
            print(
                f"Rotation '%s' already computed (skipping optimization)." %
                name)

        if rotation not in self.loadings:
            self.loadings[rotation] = np.matmul(
                self.loadings[Rotations.ORIGINAL], self.rot_mats[rotation]
            )

        self.cur_rotation = rotation

        return

    def continue_optimization(self, rotation, max_iter=50):
        """
        Continues an optimization which got interupted before. For this the self.dstats 
        have to be set. 

        Parameters:
        -----------
        rotation  :  Rotations     Rotation criterion to use
        max_iter  :  int           Number of iterations to optimize (default: 50)
        """

        assert rotation in self.rot_mats, f"rotation  {rotation} not computed yet"
        assert (
            rotation is not Rotations.ORIGINAL and rotation is not Rotations.PCA
        ), "Not possible for PCA and ORIGINAL"

        for i in range(len(self.dstats[rotation])):

            init_rot_mat = self.dstats[rotation][i]["rotation_matrix"]

            if rotation.name.startswith("FP"):
                self.rotator.toggle_use_fp(True, mean_logits=self.mean)
            else:
                self.rotator.toggle_use_fp(False, mean_logits=self.mean)

            _, rot_mat, sub_dstats = self.rotation_funcs[rotation](
                self.loadings[Rotations.ORIGINAL],
                init_rot_mat=init_rot_mat,
                max_iter=max_iter,
            )

            self.dstats[rotation][i]["rotation_matrix"] = rot_mat
            self.dstats[rotation][i]["obj"] += sub_dstats["obj"]
            self.dstats[rotation][i]["ls_steps"] += sub_dstats["ls_steps"]
            self.dstats[rotation][i]["it_times"] += sub_dstats["it_times"]
            self.dstats[rotation][i]["s"] += sub_dstats["s"]

            obj_best = self.dstats[rotation][self.dstats["best_runs"]
                                             [rotation]]["obj"][-1]
            obj_new = sub_dstats["obj"][-1]

            if obj_best > obj_new:
                print(
                    f"Rotation improved from {obj_best} to {obj_new} in {max_iter} iterations!"
                )
                self.dstats["best_runs"][rotation] = i

                self.rot_mats[rotation] = rot_mat
                self.loadings[rotation] = np.matmul(
                    self.loadings[Rotations.ORIGINAL], self.rot_mats[rotation]
                )

    def get_fps_full(
        self,
        m=250,
        plot=True,
        sfprobs=False,
        full_psi=True,
        target_folder="./plots",
        scale=2,
    ):
        """compute Monte Carlo approximation of flow probabilities for the full
        factor model

        m:             int     number of MC samples
        plot:          bool    plot the full flow probabilities
        sfprobs:       bool    compute softmax FPs (not used in paper)
        full_psi:      bool    include the model's diagonal into the computation
        target_folder: bool    target folder for the plots
        scale:         bool    scale the flow probabilities for better visualization
        """

        os.makedirs(target_folder, exist_ok=True)
        loading_mat = self.get_loadings(self.cur_rotation, sort=True)

        def e(x): return np.argmax(
            rearrange(x, "(p c) -> p c", c=self.num_classes), axis=-1
        )

        _idx_spatial = np.arange(self.h * self.w)
        mean = rearrange(self.mean, "(p c) -> p c", c=self.num_classes)
        p_full = np.zeros_like(mean, dtype=np.float32)

        if full_psi:
            p_full_psi = np.zeros_like(mean, dtype=np.float32)
        sp_full_psi = np.zeros_like(mean)
        mean_pred = np.zeros_like(mean)
        mean_pred[_idx_spatial, e(self.mean)] = 1
        #        print("mindiag", np.min(self.sqdiag), np.max(self.sqdiag))
        for i in range(m):
            z = np.random.randn(self.rank)

            logits = self.mean + np.dot(loading_mat, z)

            p_full[_idx_spatial, e(logits)] += 1

            if full_psi:
                eps = np.random.randn(len(loading_mat))

                logits_psi = logits + self.sqdiag * eps
                p_full_psi[_idx_spatial, e(logits_psi)] += 1
            #            logits_r = rearrange(logits_psi, "(p c) -> p c", c=self.num_classes)
            logits_r = rearrange(logits, "(p c) -> p c", c=self.num_classes)

            if sfprobs:
                sp_full_psi += softmax(logits_r, axis=1)  # expit = sigmoid

        p_full /= m
        f_full = p_full - mean_pred

        if full_psi:
            p_full_psi /= m
            f_full_psi = p_full_psi - mean_pred
            sp_full_psi /= m
            if sfprobs:
                sf_full_psi = sp_full_psi - softmax(mean, axis=1)

        if plot:
            plot_leg = 0
            axfontsize = 36
            plot_sample = self.sample is not None
            plot_gt = self.labels is not None
            plot_entropy = 1
            plot_sentropy = 1
            plot_sfp = 1
            plot_fp = 1
            plot_fpnopsi = 0
            plot_conf = 1
            ncols = (
                1
                + int(plot_sample)
                + int(plot_gt)
                + int(plot_entropy)
                + int(plot_sfp)
                + int(plot_fpnopsi)
                + int(plot_conf)
                + int(plot_sentropy)
                + int(plot_fp)
            )
            fig, axs = plt.subplots(
                ncols=ncols, figsize=(self.h / self.w * (ncols * 3 + 2), 4)
            )
            i = 0
            if plot_sample:
                axs[i].set_title("Image", fontsize=axfontsize)
                axs[i].imshow(self.sample)
                i += 1
            if plot_gt:
                axs[i].set_title("Ground truth", fontsize=axfontsize)

                if len(self.labels) == 4:
                    labels = self.cmap(
                        np.concatenate(
                            [
                                np.concatenate(
                                    [
                                        self.labels[0],
                                        np.ones([self.w, 3]),
                                        self.labels[1],
                                    ],
                                    1,
                                ),
                                np.ones([3, 2 * self.h + 3]),
                                np.concatenate(
                                    [
                                        self.labels[2],
                                        np.ones([self.w, 3]),
                                        self.labels[3],
                                    ],
                                    1,
                                ),
                            ],
                            0,
                        )
                    )
                else:
                    labels = np.mean(
                        [self.cmap(l.astype(np.int64)) for l in self.labels], 0
                    )
                axs[i].imshow(labels)
                i += 1
            axs[i].set_title("Mean pred.", fontsize=axfontsize)
            pred = rearrange(
                self.mean, "(w h c) -> w h c", w=self.w, c=self.num_classes
            )
            pred = np.argmax(pred, axis=2)  # / self.num_classes
            pred = self.cmap(pred)
            axs[i].imshow(pred)
            i += 1
            if plot_fp:
                fp = rearrange(f_full_psi, "(w h) c -> w h c", w=self.w)
                img = self.get_uncertainty_image(fp, upscale=scale)
                #                axs[i].set_title("FP full (L1=%.1f)"%np.linalg.norm(f_full_psi, 1),
                #                   fontsize=axfontsize)
                axs[i].set_title("FP full", fontsize=axfontsize)
                axs[i].imshow(img)
                i += 1
            #                if plot_fpnopsi:
            #                    fp =  rearrange(f_full, "(w h) c -> w h c", w=self.w)
            #                    img = self.get_uncertainty_image(fp, upscale=2)
            #                    axs[i].set_title("FP full-Psi", fontsize=axfontsize)
            #                    axs[i].imshow(img)
            #                    i += 1
            if sfprobs and plot_sfp:
                fp = rearrange(sf_full_psi, "(w h) c -> w h c", w=self.w)
                img = self.get_uncertainty_image(fp, upscale=scale * 2)
                axs[i].set_title("S-FP full", fontsize=axfontsize)
                axs[i].imshow(img)
                i += 1

            if plot_conf:
                axs[i].set_title("Confidence", fontsize=axfontsize)
                conf = np.max(sp_full_psi, axis=1)
                conf = rearrange(conf, "(w h) -> w h", w=self.w)
                axs[i].imshow(conf, cmap=cm.Greys_r, vmin=0, vmax=1)
                i += 1
            if plot_entropy:
                axs[i].set_title("Entropy", fontsize=axfontsize)
                #                mask
                ent = np.multiply(p_full_psi, np.log(p_full_psi))
                ent = np.nan_to_num(ent, nan=0.0, posinf=None, neginf=None)
                ent = -np.sum(ent, axis=1) / np.log2(10)
                ent = rearrange(ent, "(w h) -> w h", w=self.w)

                #                axs[i].imshow(ent, cmap=cm.Greys_r)
                axs[i].imshow(ent, cmap=cm.Greys_r)  # , vmin=0, vmax=1)

                #                    axs[i].imshow(ent, cmap=cm.jet)
                i += 1
            if plot_sentropy:
                axs[i].set_title("S-Entropy", fontsize=axfontsize)
                ent = np.multiply(sp_full_psi, np.log(sp_full_psi))
                ent = np.nan_to_num(ent, nan=0.0, posinf=None, neginf=None)
                ent = -np.sum(ent, axis=1) / np.log2(10)

                ent = rearrange(ent, "(w h) -> w h", w=self.w)
                #                axs[i].imshow(ent, cmap=cm.Greys_r)
                axs[i].imshow(ent, cmap=cm.Greys_r, vmin=0, vmax=1)
                #                    axs[i].imshow(ent, cmap=cm.jet)
                i += 1
            # TODO: following legend is hard-coded for SEN12MS
            DFC_classes = [
                "Forest",
                "Shrubland",
                "Savanna",
                "Grassland",
                "Wetlands",
                "Croplands",
                "Urban/Built-up",
                "Snow/Ice",
                "Barren",
                "Water",
            ]
            if plot_leg:
                import matplotlib.patches as mpatches

                lgs = []
                for i in range(len(DFC_classes)):
                    lgs.append(
                        mpatches.Patch(
                            color=self.cmap(
                                i / self.num_classes),
                            label=DFC_classes[i]))
                axs[-1].legend(
                    handles=lgs,
                    fontsize=axfontsize - 7,
                    bbox_to_anchor=(1, 1.034),
                    loc="upper left",
                )

            for i in range(ncols):
                axs[i].xaxis.set_major_locator(ticker.NullLocator())
                axs[i].yaxis.set_major_locator(ticker.NullLocator())
            plt.tight_layout()
            if isinstance(self.sid, type(1)) or isinstance(self.sid, np.int64):
                plt.savefig(
                    os.path.join(
                        target_folder,
                        "flowprobs_full_{}_scale{}.pdf".format(
                            self.sid,
                            scale),
                    ))
            else:
                sid = self.sid.replace(".tif", "")
                plt.savefig(
                    os.path.join(
                        target_folder,
                        "flowprobs_full_{}_scale{}.pdf".format(sid, scale),
                    )
                )
            plt.show()

            plt.close()

        if full_psi:
            return f_full, f_full_psi
        else:
            return f_full

    def plot_samples_stepwise_factor(
        self, id_list=None, sort=True, filename=None, sformat="png"
    ):
        """
        Computes and prints the samples for deterministic latent variables for individual factors. 

        Parameters: 
        -----------
        id_list:     list     ids of factors to use
        sort:        bool     sort the factors by flow probabily norm
        filename:    str      path to save the plots to
        """

        if id_list is None:
            id_list = np.arange(self.rank)

        if self.num_classes <= 2:
            cmap = cm.Greys
        else:
            assert self.num_classes == 10, "Implemented only for SEN12MS"
            cmap = self.cmap

        fps = self.get_flow_probabilities(self.cur_rotation, sort=sort)
        loadings = self.get_loadings(self.cur_rotation, sort=sort)
        idx_list = self.sorted_idx

        nrows = len(id_list)
        ncols = 10
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(2 * ncols + 1, 2.1 * nrows)
        )
        for ii, zid in enumerate(id_list):
            z = np.zeros(self.rank)
            axs[ii, 0].set_ylabel("F%d" % (zid), fontsize=20)
            for i in range(ncols):
                idx = idx_list[zid]
                #            eps = np.random.randn(self.flatshape)
                zval = i - int(ncols / 2)
                eta = self.mean + loadings[:, idx] * zval
                eta = rearrange(eta, "(s c) -> s c", c=self.num_classes)
                if self.num_classes == 2:
                    sample = (eta[:, 1] - eta[:, 0]) > 0
                    sample = rearrange(sample, "(h w) -> h w", h=self.h)
                elif self.num_classes == 1:
                    sample = eta > 0
                    sample = np.squeeze(sample)
                    sample = rearrange(sample, "(h w) -> h w", h=self.h)
                else:
                    sample = rearrange(
                        np.argmax(eta, axis=-1), "(h w) -> h w", h=self.h
                    )
                    sample = sample / self.num_classes
                    sample = self.cmap(sample)  # [:, :, 0:3] # TODO: simplify?

                axs[ii, i].imshow(sample, vmin=0, vmax=1, cmap=cmap)
                #            axs[i].set_title("z%d=%.2f"%(zid, z[zid]))
                if ii == 0:
                    axs[ii, i].set_title("z=%d" % (zval), fontsize=20)
                axs[ii, i].xaxis.set_major_locator(ticker.NullLocator())
                axs[ii, i].yaxis.set_major_locator(ticker.NullLocator())
        plt.tight_layout()
        if filename is None:
            filename = "Plots/sample%s_stepwise.%s" % (str(self.sid), sformat)
        plt.savefig(filename)
        plt.close()

    def plot_single_steps(
        self,
        id_list=None,
        file_prefix=None,
        sformat="png",
        step_list=[-1.0, -0.5, 0.0, 0.5, 1.0],
        target_folder="./plots",
    ):
        """
        Inspect samples from steps for a single component of a given factor model

        Parameters: 
        -----------
        id_list:       list   ids of factors to use
        file_prefix    str    prefix for saved images
        sformat        str    save format ["png", "pdf"]
        step_list      list   list of steps to visualize. (default: [-1,-0.5,0,0.5,1])
        target_folder  str    folder to save images into
        """

        os.makedirs(target_folder, exist_ok=True)

        if id_list is None:
            id_list = np.arange(self.rank)

        _ = self.get_flow_probabilities(
            self.cur_rotation, sort=False)  # to get sorted
        loadings = self.get_loadings(self.cur_rotation, sort=False)

        n_cols = 2 + len(step_list)
        fpos, fneg = self.get_onesidedflowprobs(rotations=self.cur_rotation)

        for ii, zid in enumerate(id_list):

            fig, axs = plt.subplots(
                nrows=1,
                ncols=n_cols,
                figsize=(1.5 + 4 * len(step_list), 3.4),
                gridspec_kw={"width_ratios": [1] + len(step_list) * [0.8] + [1]},
            )

            for iii, zval in enumerate(step_list):

                eta = self.mean + loadings[:, self.sorted_idx[zid]] * zval
                eta = rearrange(eta, "(s c) -> s c", c=self.num_logits)

                sample = rearrange(
                    np.argmax(
                        eta,
                        axis=-1),
                    "(h w) -> h w",
                    h=self.h)
                sample = self.cmap(sample)

                axs[iii + 1].imshow(sample)
                axs[iii + 1].axis("off")
                # axs[iii+1].set_anchor('S')

            axs[0].imshow(fneg[zid])
            axs[0].tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                left=False,
                labelleft=False,
            )  # labels along the bottom edge are off
            axs[0].set_anchor("S")
            axs[0].set_ylabel("Factor {}".format(zid + 1), fontsize=42)

            axs[-1].imshow(fpos[zid])
            axs[-1].tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                left=False,
                labelleft=False,
            )  # labels along the bottom edge are off
            axs[-1].set_anchor("S")

            plt.subplots_adjust(wspace=0.001)
            fig.tight_layout()
            fig.patch.set_linewidth(5)
            fig.patch.set_edgecolor("black")
            plt.savefig(
                os.path.join(
                    target_folder,
                    file_prefix +
                    "_factor_{}_full".format(zid) +
                    ".{}".format(sformat),
                ),
                edgecolor=fig.get_edgecolor(),
            )
            plt.close()

    def plots_statistics(self, filename=None):
        """
        Plots the objective vs. iterations for the current rotation.
        The training statistics have to be stored in self.dstats to use
        this method. 

        Parameter:
        ----------
        filename  :  str     Path to file where to save the plot
        """

        if (
            self.cur_rotation in self.dstats
            and self.dstats[self.cur_rotation] is not None
        ):
            plt.figure()
            plt.plot(self.dstats[self.cur_rotation]["obj"])
            plt.title("Objective Value for " + self.cur_rotation.name)
            plt.savefig(filename)
            plt.close

    def get_uncertainty_image(self, flowprobs, upscale=2):
        """
        Generates a visualization of given flow probabilities
        
        Parameters: 
        -----------
        flowprobs  :  nparray   flow probabilities
        upscale    :  int       upscaling factor for better visualization
        """
        img = np.zeros((self.w, self.h, 4))
        img[:, :, -1] = 1
        weights = np.zeros((self.w, self.h))
        for c in range(self.num_classes):
            factor_j = flowprobs[:, :, c]
            mask = factor_j > 0.01
            weights[mask] += factor_j[mask]
            if np.sum(mask) > 0:
                color = self.cmap(c)
                for x in range(3):
                    img[mask, x] += upscale * factor_j[mask] * color[x]

        return img

    def compute_fac_visualization(self, rotation=None):
        """
        Generates a visualization of the flow probabilities of the individual 
        factors and returns it. 
        
        Parameters: 
        -----------
        rotation  :  Rotations     Which rotation criterion to visualize. 
        """

        rotation = self.cur_rotation if rotation is None else rotation
        num_c = self.num_classes
        fps = self.get_flow_probabilities(rotation, sort=True)

        # reshape for plotting
        fps = rearrange(fps, "(w h c) r -> w h c r", w=self.w, c=num_c)
        factor_imgs = []

        for j in range(self.rank):
            factor_img = np.zeros((self.w, self.h, 4))
            factor_img[:, :, -1] = 1

            for c in range(num_c):
                factor_j = fps[:, :, c, j]
                mask = factor_j > 0.05
                if np.sum(mask) > 0:
                    c_image = np.zeros((self.w, self.h, 4))
                    color = self.cmap([c / self.num_classes])
                    c_image[mask, :] = color
                    for x in range(4):
                        factor_img[mask, x] = np.multiply(
                            2 * factor_j[mask], c_image[mask, x]
                        ) + np.multiply(1 - 2 * factor_j[mask], factor_img[mask, x])

            factor_imgs.append(factor_img)

        return factor_imgs

    def visualize_fac(
        self,
        rotation=None,  # plot_meanpred=False,
        filename=None,
        sformat="png",
        upscale=2,
        fchar="f",
        target_folder="./plots",
    ):
        """
        Generates a visualization of the factor specific FPs individual factors and 
        saves it to a file. 
        
        Parameters: 
        -----------
        rotation  :  Rotations     Which rotation criterion to visualize. 
        filename  :  str           Name of saved file
        sformat   :  str           Format of saved file ["png","pdf"]
        upscale   :  int           Upscale factor for better visualization
        fchar     :  str           Two-sided ("F") or one-sided negative ("Fneg")
                                   or one-sided positive ("Fpos") flow probabilities.
        """

        num_c = self.num_classes
        os.makedirs(target_folder, exist_ok=True)

        if rotation is None:
            rotations = [r for r in self.rot_mats]
        else:
            rotations = rotation

        # Plotting
        axfontsize = 20
        ncols = self.rank
        nrows = len(rotations)
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, 2.2 * nrows))
        if nrows == 1:
            axs = np.expand_dims(axs, 0)

        for i, rot in enumerate(rotations):
            rotname = (
                rot.name.replace("Rotations.", "")
                .replace("FP", "FP-")
                .replace("ORIGINAL", "UNROTATED")
            )
            rotname += "" if fchar == "f" else "(%s)" % fchar
            axs[i, 0].set_ylabel("%s" % rotname, fontsize=axfontsize - 3)

            if fchar == "f":
                fps = self.get_flow_probabilities(rot, sort=True)
            else:
                loadings = self.get_loadings(rot, sort=False)
                fps_dict = get_flowprobs(
                    self.mean,
                    loadings,
                    self.num_logits,
                    usepool=False,
                    gradient=False,
                    onesided=True,
                )
                fps = fps_dict[fchar]
            fps = rearrange(fps, "(w h c) r -> w h c r", w=self.w, c=num_c)

            factor_imgs = []
            for j in range(self.rank):
                fps_j = fps[:, :, :, j]
                factor_img = self.get_uncertainty_image(fps_j, upscale=upscale)
                if i == 0:
                    fps_j = rearrange(fps_j, "w h c -> (w h) c")
                    axs[i, j].set_title("Factor {}".format(
                        j + 1), fontsize=axfontsize)

                factor_imgs.append(factor_img)
                axs[i, j].imshow(factor_img, vmin=0, vmax=1)

            for j in range(ncols):
                axs[i, j].xaxis.set_major_locator(ticker.NullLocator())
                axs[i, j].yaxis.set_major_locator(ticker.NullLocator())

        plt.tight_layout()
        if filename is None:
            filename = os.path.join(
                target_folder,
                "Sample%s_multirot%s.%s" % (str(self.sid), fchar, sformat),
            )

        os.makedirs(os.path.join(*filename.split("/")[:-1]), exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
        #        plt.close()
        return factor_imgs

    def get_onesidedflowprobs(self, rotations=None, upscale=2):
        """
        Computes one-sided flow probabilities (positive and negative) 

        Parameters: 
        -----------
        rotation  :  Rotations     Rotations of which criterion to use.
        upscale   :  int           upscaling for better visualization
        """
        
        num_c = self.num_classes
        if rotations is None:
            rotations = [r for r in self.rot_mats]
        elif not isinstance(rotations, list):
            rotations = [rotations]

        for i, rot in enumerate(rotations):
            loadings = self.get_loadings(rot, sort=False)
            fps_dict = get_flowprobs(
                self.mean,
                loadings,
                self.num_logits,
                usepool=False,
                gradient=False,
                onesided=True,
            )
            fpos = rearrange(
                fps_dict["fpos"], "(w h c) r -> w h c r", w=self.w, c=num_c
            )
            fneg = rearrange(
                fps_dict["fneg"], "(w h c) r -> w h c r", w=self.w, c=num_c
            )

            self.sorted_idx = get_sortidx_bynorm(fps_dict["f"])

            fpos_imgs = []
            fneg_imgs = []
            for j in self.sorted_idx:
                img = self.get_uncertainty_image(
                    fpos[:, :, :, j], upscale=upscale)
                fpos_imgs.append(img)
                img = self.get_uncertainty_image(
                    fneg[:, :, :, j], upscale=upscale)
                fneg_imgs.append(img)

        return fpos_imgs, fneg_imgs

    def visualize_factors(
        self,
        lim=10,
        maxfactors=10,
        filename=None,
        show=True,
        plot_title=True,
        plot_logits=None,
        sformat="png",
        omit_bg_class=0,
        target_folder="./plots",
    ):
        """
        Generates a visualization of the individual factors and saves it to a file. 
        
        Parameters: 
        -----------
        lim            :   int    minimum and maximum intensity to visualize [-lim, lim]
        maxfactor      :   int    number of factors to show at most
        filename       :   str    name of save file
        show           :   bool   show the generated plot
        plot_title     :   bool   add title to plot
        sformat        :   str    save format ["png","pdf"]
        omit_bg_class  :   int    omits the background class from the plot
        target_folder  :   str    folder to save the images to
        """

        os.makedirs(target_folder, exist_ok=True)

        if plot_logits is None:
            plot_logits = self.num_classes <= 2
        if self.num_classes == 2:
            omit_bg_class = 1
        mean = self.mean
        loading_mat = self.get_loadings(rotation=self.cur_rotation, sort=False)
        num_c = self.num_classes
        k = min(self.rank, maxfactors)
        loading_mat = rearrange(
            loading_mat,
            "(p c) r -> p c r",
            c=self.num_classes)
        assert self.w * \
            self.h == loading_mat.shape[0], "logits for each class given?"
        mean = rearrange(mean, "(p c) -> p c", c=self.num_classes)

        # subtract logits from first (bg) class from all other classes
        mean = mean.copy()
        loading_mat = loading_mat.copy()
        for i in range(1, self.num_classes):
            mean[:, i] -= mean[:, 0]
            for j in range(k):
                loading_mat[:, i, j] -= loading_mat[:, 0, j]
        for j in range(k):
            loading_mat[:, 0, j] = 0
        mean[:, 0] = 0

        self.get_flow_probabilities(self.cur_rotation, sort=True)
        fps = self.fps[self.cur_rotation]
        # reshape for plotting
        imgloading_mat = rearrange(
            loading_mat, "(w h) c r -> w h c r", w=self.w)
        fps = rearrange(fps, "(w h c) r -> w h c r", w=self.w, c=num_c)
        imgmean = rearrange(mean, "(w h) c -> w h c", w=self.w)

        # Plotting
        scaling = 1
        row_fac = 1 + int(plot_logits)
        nrows = row_fac * (num_c - omit_bg_class)
        ncols = k + 1
        figsize = (maxfactors + 1) * 2, (row_fac * num_c - row_fac) * 2
        if maxfactors < self.rank:
            scaling = 10 / (maxfactors + 1)
            figsize = (10, (row_fac * num_c - row_fac) * scaling)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        cmap = cm.seismic
        axfontsize = 12 * scaling * 1.8
        axs[0, 0].set_title("mean", fontsize=axfontsize)
        for c in range(omit_bg_class, num_c):
            b = c - omit_bg_class
            if plot_logits:
                a = row_fac * (c - omit_bg_class)
                b = a + 1
                axs[a, 0].set_ylabel("Logits [%d]" %
                                     c, fontsize=axfontsize - 2)
                axs[a, 0].imshow(imgmean[:, :, c], cmap=cmap,
                                 vmin=-lim, vmax=lim)
            axs[b, 0].set_ylabel("FP [%d]" % c, fontsize=axfontsize - 2)
            pred = c == np.argmax(imgmean, axis=2)
            #            color = self.cmap()
            #            pred = self.cmap(pred)[:, :, 0:3]
            axs[b, 0].imshow(pred, cmap=cm.Greys, vmin=0, vmax=1)
        for i in range(k):
            ii = self.sorted_idx[i]
            axs[0, i + 1].set_title(r"$z_%d$" % i, fontsize=axfontsize)
            for c in range(omit_bg_class, num_c):
                b = c - omit_bg_class
                if plot_logits:
                    a = row_fac * (c - omit_bg_class)
                    b = a + 1
                    factor_ic = imgloading_mat[:, :, c, ii]
                    axs[a, i + 1].imshow(factor_ic, cmap=cmap,
                                         vmin=-lim, vmax=lim)
                axs[b, i + 1].imshow(fps[:, :, c, ii],
                                     cmap=cm.bwr, vmin=-1, vmax=1)

        for i in range(nrows):
            for j in range(ncols):
                axs[i, j].xaxis.set_major_locator(ticker.NullLocator())
                axs[i, j].yaxis.set_major_locator(ticker.NullLocator())
        if plot_title:
            plt.suptitle(self.cur_rotation.name, y=1.09, fontsize=25)
        plt.tight_layout()
        #        plt.subplots_adjust(top=0.85)
        if filename is None:
            filename = os.path.join(
                target_folder, "Sample{}_{}.{}".format(
                    self.sid, self.cur_rotation.name, sformat), )

        plt.savefig(filename, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

def visualize_examples(
    dataset,
    data_path,
    target_dir,
    plot_full_fps=True,
    plot_steps=True,
    plot_factors=True,
    ):
    """
    This method loads multiple pickle files and visualizes the content into multiple files. 

    Parameters:
    -----------
    dataset        str    which dataset to use ["LIDC", "SEN12MS", "CamVid"]
    data_path      str    path to folder containing sample pickle files
    target_dir     str    target folder for visualizations
    plot_full_fps  bool   visualize full flow probabilities
    plot_steps     bool   visualize deterministic steps for single factors
    plot_factors   bool   visualize the factors
    """

    np.seterr(
        divide="ignore"
    )  # ignore zero division errors (potentially in flow prob computation)
    np.random.seed(100)

    assert dataset in ["LIDC", "SEN12MS", "CamVid"]

    if isinstance(data_path, list):
        filenames = data_path
    elif data_path.ends_with(".pkl"):
        filenames = [data_path]
    else:
        filenames = [
            os.path.join(data_path, p)
            for p in os.listdir(data_path)
            if p.endswith(".pkl")
        ]

    for filename in filenames:
        if dataset == "CamVid":

            DSET_MEAN = np.array(
                [[[0.41189489566336, 0.4251328133025, 0.4326707089857]]]
            )
            DSET_STD = np.array(
                [[[0.27413549931506, 0.28506257482912, 0.28284674400252]]]
            )

            with open(filename, "rb") as f:
                dist = pickle.load(f)
            print("File %s" % filename)

            loading_mat = dist["cov_factor"]
            mean = dist["mean"].flatten()
            diag = dist["cov_diag"].flatten()
            sampleid = dist["sample_id"]
            sample = np.transpose(
                dist["sample"], [
                    1, 2, 0]) * DSET_STD + DSET_MEAN
            labels = [dist["labels"]]

            shape = 360, 480  # w/h, num_classes

            fmodel = FactorModel(
                mean,
                loading_mat,
                diag,
                data_shape=shape,
                cmap="CamVid",
                sid=sampleid,
                num_classes=11,
                sample=sample,
                labels=labels,
            )

        elif dataset == "LIDC":
            with open(filename, "rb") as f:
                dist = pickle.load(f)
            print("File %s" % filename)

            try:
                loading_mat = dist["Factor_flat"]
                mean = dist["Mean_flat"].flatten()
                diag = dist["diag_flat"].flatten()
                sample = None
                labels = None
            except BaseException:
                loading_mat = dist["cov_factor"]
                mean = dist["mean"].flatten()
                diag = dist["cov_diag"].flatten()
                sampleid = dist["sample_id"]
                sample = dist["sample"]
                labels = np.transpose(dist["labels"], [2, 0, 1])

            print("norm", np.linalg.norm(diag) / (128 ** 2 * 2))
            shape = 128, 128  # w/h, num_classes

            fmodel = FactorModel(
                mean,
                loading_mat,
                diag,
                data_shape=shape,
                sid=sampleid,
                num_classes=2,
                sample=sample,
                labels=labels,
                cmap="LIDC",
            )

        elif dataset == "SEN12MS":
            with open(filename, "rb") as f:
                dist = pickle.load(f)

            print("File %s" % filename)

            mean = dist["mean"]
            try:
                loading_mat = dist["factor"]
                diag = dist["diag"].flatten()
            except BaseException:
                loading_mat = dist["cov_factor"]
                diag = dist["cov_diag"].flatten()
            sampleid = dist["sample_id"]
            shape = 256, 256  # w/h, num_classes
            num_classes = 10

            sample = np.transpose(
                dist["sample"][[2, 1, 0], :, :], axes=(1, 2, 0))
            sample = np.clip(sample * 10000, 0, 2000) / 2000

            fmodel = FactorModel(
                mean,
                loading_mat,
                diag,
                data_shape=shape,
                sid=sampleid,
                cmap="SEN12MS",
                sample=sample,
                labels=dist["labels"],
                num_classes=num_classes,
            )

        # do visualization and save
        k_list = list(dist["rot_mats"].keys())
        for r in k_list:
            new_key = cast_rotations(r)

            # hack to fix rotation enum reference
            if r is not new_key:
                dist["rot_mats"][new_key] = dist["rot_mats"][r]
                del dist["rot_mats"][r]

        fmodel.add_rotations_from_dict(dist)

        if plot_full_fps:
            t_path = os.path.join(target_dir, "fps_full")
            os.makedirs(t_path, exists_ok=True)
            fmodel.get_fps_full(
                plot=True,
                sfprobs=True,
                target_folder=t_path,
                scale=s)

        if plot_factors:
            t_path = os.path.join(target_dir, "factors")
            os.makedirs(t_path, exists_ok=True)
            fmodel.visualize_fac(
                fchar="f",
                target_folder=t_path,
                sformat="pdf")

        if plot_steps:
            t_path = os.path.join(target_dir, "steps")
            for r in fmodel.rot_mats:
                fmodel.rotate(r)
                fmodel.plot_single_steps(
                    id_list=None,
                    step_list=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    target_folder=t_path,
                    file_prefix="{}_{}".format(
                        filename.split("/")[-1].replace(".pkl", ""), r.name
                    ),
                    sformat="pdf",
                )

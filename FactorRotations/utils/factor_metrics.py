"""
This script contains a number of metrics to evaluate either segmentation performance or 
different rotations and corresponding loadings and flow probability representations.

For details please read the description of the single methods.
"""

import itertools
import numpy as np


def IoU(sample1, sample2, num_classes=None, dim=None, reduce=True):
    """
    Intersection over Union. For usage as done by Monteiro et al.'s work on stochastic segmentation.

    Parameters
    ----------
    samples1 : list or nparr     sample(s)
    samples2 : list or nparr     compare sample(s)
    num_classes : int            number of classes, needed if labels are not one-hot-encoded
    dim : int                    dimension of prediction/one-hot vector, needed if sample is not np.int64
    reduce :                     average over all samples in list (default: True)
    average_classes :            average over IoU of singler classes
    """

    if not isinstance(np.max(sample1), np.int64):
        if dim is None:
            raise Exception(
                "Give the data in long format or pass a dimension parameter!"
            )
        if num_classes is None:
            num_classes = sample1.shape[dim]
        if sample1.shape[dim] == 1:
            sample1 = np.squeeze(np.round(sample1), axis=dim).astype(np.int64)
        else:
            sample1 = np.argmax(sample1, dim)

    if not isinstance(np.max(sample2), np.int64):
        if dim is None:
            raise Exception(
                "Give the data in int64 format or pass a dimension parameter!"
            )
        if num_classes is None:
            num_classes = sample2.shape[dim]
        if sample2.shape[dim] == 1:
            sample2 = np.squeeze(np.round(sample2), axis=dim).astype(np.int64)
        else:
            sample2 = np.argmax(sample2, dim)

    if num_classes is None:
        raise Exception(
            "For labels given in int64 format also pass the total number of classes."
        )

    class_wise_iou = []
    if num_classes == 1:
        start = 1
        num_classes = 2
    else:
        start = 0

    reject_class = num_classes

    for class_no in np.arange(start, num_classes):
        mask1 = sample1 == class_no
        mask2 = sample2 == class_no

        reject = np.sum((sample2 == reject_class) * (sample1 == class_no)) + \
            np.sum((sample1 == reject_class) * (sample2 == class_no))

        mask_intersect = np.sum(mask1 * mask2)
        mask_union = np.sum(mask1) + np.sum(mask2) - reject - mask_intersect

        r = np.where(
            mask_union < 0.5,
            np.ones_like(mask_union),
            mask_intersect /
            mask_union)

        class_wise_iou.append(r)

    assert np.all(1 >= np.mean(class_wise_iou) >= 0)
    if reduce:
        return np.mean(class_wise_iou)
    else:
        return np.mean(class_wise_iou, axis=0)


def sample_diversity(samples, dist_fcn=IoU, dim=None, num_classes=None):
    """
    Compute the average pairwise distance for a given list of samples. 
    The distance is computed by the passed distance function (default: IoU).

    Parameters
    ----------
    samples      :  list or nparr     samples to be compared to each other
    dist_fcn     :  method            function to be used as a distance meaure (default: IoU)
    dim          :  int[]             dimension (only needed if number of classes not given)
    num_classes  :  int               number of classes (only needed if dim not given)  
    """

    if not isinstance(np.max(samples), np.int64):
        if dim is None:
            raise Exception(
                "Give the data in int64 format or pass a dimension parameter!"
            )
        num_classes = samples.shape[dim]
        if num_classes == 1:
            samples = np.squeeze(np.round(samples)).astype(np.int64)
        else:
            samples = np.argmax(samples, dim)

    sample_ids = list(
        itertools.product(np.arange(len(samples)), np.arange(len(samples)))
    )

    d_list = list(
        map(
            lambda idx: 1
            - dist_fcn(samples[idx[0]], samples[idx[1]], num_classes=num_classes),
            sample_ids,
        )
    )

    return np.mean(d_list)


def pairwise_seperation_L1(factor_model):
    """
    Evaluates how pairwise "overlapping" the factor-wise flow probabilities of 
    a given factor model are. 
    Computes the scalar product of each combination of (absolut) loadings in a 
    given factor model. 

    Parameters
    ----------
    factor_model    :    object of class FactorModel that should be evaluated. 
    """

    loadings = factor_model.get_flow_probabilities(
        factor_model.cur_rotation, sort=True)

    loadings = loadings / np.sum(loadings ** 2, axis=-2, keepdims=True)

    sample_ids = np.add(np.triu_indices(loadings.shape[-1], 1), 0).T

    res = list(
        map(
            lambda idx: np.dot(
                np.abs(loadings[:, idx[0]]), np.abs(loadings[:, idx[1]])
            ),
            sample_ids,
        )
    )

    return res


def loadingwise_SD(factor_model):
    """
    Evaluates the sample diversity induced by the single factors of a given 
    factor model (based on the currently defined rotation in the factor model).

    Parameters
    ----------
    factor_model    :    object of class FactorModel that should be evaluated. 
    """
    
    sample_list = []
    for i in range(factor_model.rank):
        mask = np.transpose(np.eye(1, factor_model.rank, i))
        sample_list.append(
            np.argmax(
                factor_model.get_logits_from_sample(
                    num_samples=100, mask=mask, sort=True, use_diag=False
                ),
                axis=-1,
            ).astype(np.int64)
        )

    res = list(map(lambda x: sample_diversity(
        samples=x, num_classes=factor_model.num_classes), sample_list, ))
    return res


def loadingwise_L1(factor_model):
    """
    Computes the l1-norm of the factor-wise flow probabilities for a given 
    factor model and its current rotation. 

    Parameters
    ----------
    factor_model    :    object of class FactorModel that should be evaluated. 
    """

    res = np.linalg.norm(factor_model.get_flow_probabilities(
        factor_model.cur_rotation, sort=True), ord=1, axis=-2, )

    return res


def pairwise_cosine_similarity(factor_model, eps=10e-20):
    """
    Evaluates how pairwise separability of the factor-wise flow probabilities
    of a given factor model are. 
    Computes the cosine similarity of each combination of (absolut) loadings in a 
    given factor model. 

    Parameters
    ----------
    factor_model    :    object of class FactorModel that should be evaluated. 
    """

    loadings = factor_model.get_flow_probabilities(
        factor_model.cur_rotation, sort=True)

    loadings = loadings / (
        np.linalg.norm(loadings, ord=2, axis=-2, keepdims=True) + eps
    )

    sample_ids = np.add(np.triu_indices(loadings.shape[-1], 1), 0).T

    res = list(map(lambda idx: np.dot(
        loadings[:, idx[0]], loadings[:, idx[1]]), sample_ids))

    return res


def hoyer_weighted(factor_model, eps=10e-20):
    """
    Evaluates the sparsity in the rows of the matrix of the factor-wise
    flow probabilities. The flow probabilities are given by a factor model 
    and its current rotation. This evaluates the amount of factors affecting 
    the outcome of a single logit. 

    Parameters
    ----------
    factor_model    :    object of class FactorModel that should be evaluated. 
    """

    loadings = factor_model.get_flow_probabilities(
        factor_model.cur_rotation, sort=True)

    r_sqr = np.sqrt(loadings.shape[-1])

    v1 = np.round(
        np.linalg.norm(
            loadings,
            ord=1,
            axis=-1),
        decimals=20) + 1e-20
    v2 = np.round(
        np.linalg.norm(
            loadings,
            ord=2,
            axis=-1),
        decimals=20) + 1e-20

    w = v1 / (np.sum(v1, axis=-1, keepdims=True))

    return np.sum(w * (r_sqr - v1 / v2) / (r_sqr - 1))

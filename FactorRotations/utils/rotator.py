"""
Rotator class to perform various rotations of factor loading matrices.
"""
# import multiprocessing
import itertools
import multiprocessing as mp
import time

import numpy as np
from einops import rearrange
from scipy.special import ndtr
from sklearn.base import BaseEstimator


class Rotator(BaseEstimator):
    """
    The Rotator class takes a factor loading matrix and performs rotations.
    """

    def __init__(self, num_c, max_iter=100, tol=1e-5):
        """
        Initialize instance.

        Parameters
        ----------
        num_c     :  int                Number of classes in underlying segmentation task.
        max_iter  :  int,   optional    The maximum number of iterations.
        tol       :  float, optional    The convergence threshold.
        """

        self.num_c = num_c  # number of classes

        self.max_iter = max_iter
        self.tol = tol

        self.obj_val = None

        self.criterion = None  # populate with classic rotation criterion
        self.loadings = None  # original factor loadings
        self.loadings_rot = None  # loadings for rotated factors

        self.use_fp = False  # use flow probabilities?
        self.mean_logits = None  # same

        self.pool = None

    def toggle_use_fp(self, use_fp, mean_logits=None):
        """
        Configure whether flow probabilities should be used
        (if yes, mean_logits need to be provided).

        Parameters
        ----------
        use_fp       :  bool      use flow pronbabilities
        mean_logits  :  nparray   mean logits, needed for FP computation 
        """

        if use_fp:
            assert mean_logits is not None
            # TODO: assertion for shape
            self.mean_logits = mean_logits
            self.use_fp = True
        else:
            self.use_fp = False

    def get_obj_grad(self, rot_mat, gradient=True):
        """
        Compute objective (and optionally gradient) for the configuration
        that is stored in class variables.

        Parameters
        ----------
        rot_mat   :  nparray     rotation matrix
        gradient  :  bool        compute also the gradient 
        """

        # ROTATION
        self.loadings_rot = np.dot(self.loadings, rot_mat)

        # FLOW PROBS (if applicable)
        if self.use_fp:  # use flow probabilities
            fp_f_grad = get_flowprobs(
                self.mean_logits,
                self.loadings_rot,
                self.num_c,
                gradient=gradient,
                pool=self.pool,
            )
            loadings_for_criterion = fp_f_grad["f"]
        else:
            loadings_for_criterion = self.loadings_rot

        # CLASSIC ROTATION CRITERION
        obj_f_grad = self.criterion(loadings_for_criterion, gradient=gradient)

        # COMPUTE GRADIENT (if gradient=True)
        if gradient:
            grad_criterion = obj_f_grad["grad"]
            if self.use_fp:
                # compute DF(Gamma O) Dq(F(Gamma O)) - notation see the doc
                grad_fp = fp_f_grad["grad"]  # npixels x rank x num_c x num_c
                npixels, rank, num_c, _ = grad_fp.shape
                grad_criterion = rearrange(
                    grad_criterion, "(p c) k -> p c k ", c=num_c)
                grad_combined = np.empty(grad_criterion.shape)
                for p in range(npixels):
                    for j in range(rank):
                        grad_combined[p, :, j] = np.dot(
                            grad_fp[p, j, :, :], grad_criterion[p, :, j]
                        )
                grad_combined = rearrange(grad_combined, "p c k -> (p c) k ")

            else:
                grad_combined = grad_criterion

            grad = np.dot(self.loadings.T, grad_combined)

            return {"f": obj_f_grad["f"], "grad": grad}
        return {"f": obj_f_grad["f"]}

    def create_pool(self):
        """
        Build pool for parallel computing of single factors. 
        """
        self.pool = mp.Pool(mp.cpu_count())

    def solve_orthogonal(
        self,
        loadings,
        method,
        init=None,
        verb=0,
        max_iter=None,
        lr=0.1,
        tol=1e-7,
        twowaybacktracking=True,
        residual_conv=True,
        objbelowtol_conv=True,
    ):
        """
        A generic function for performing orthogonal rotations.

        Parameters
        ----------
        loadings : numpy array
            The loading matrix
        method : str
            The orthogonal rotation method to use.
        lr: float (positive)
            default learning rate
        max_iter: int
            maximum number of iterations
        twowaybacktracking: bool
            if true, use two-way backtracking in Armijo Goldstein Line Search
        allow_residual_conv: bool
            if true, use residual convergence criterion (needs calibration of self.tol)

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        rotation_mtx : numpy array, shape (n_factors, n_factors)
            The rotation matrix
        """
        arr = loadings.copy()
        n_rows, n_cols = arr.shape

        if method == "equamax":
            self.criterion = lambda l, **kwargs: self._alpha_cf_obj(
                l, kappa=n_cols / 2 / n_rows, **kwargs
            )
        if method == "quartimax":
            self.criterion = lambda l, **kwargs: self._alpha_cf_obj(
                l, kappa=0, **kwargs
            )
        elif method == "varimax":
            #            objective = lambda l: self._varimax_obj(l, minimize=True)
            self.criterion = lambda l, **kwargs: self._alpha_cf_obj(
                l, kappa=1 / n_rows, **kwargs
            )
        #            objective = lambda l: self._cf_obj(l, kappa=1) # parsimony

        # initialize the rotation matrix
        if init is None:
            rotation_matrix = np.eye(n_cols)  # TODO: this may fail
        else:
            rotation_matrix = init

        if max_iter is None:
            max_iter = self.max_iter

        if self.pool is None and self.use_fp:
            self.create_pool()

        self.loadings = arr

        d = 0  # for termination criterion
        if verb > 1:
            method_str = "%s%s" % ("FP-" if self.use_fp else "", method)
            print("Starting optimization for %s..." % method_str)
            rstr = "it=%2d ** s=%.2f\tobj=%.3f"
            print("It.\tResidual\tObj")

        # compare GPForth routine from Bernaards et al. (2005), p.18
        dstats = {
            "obj": [],
            "s": [],
            "ls_steps": [],
            "msg": None,
            "it_times": []}
        for i in range(max_iter + 1):
            tic = time.time()
            f_grad = self.get_obj_grad(rotation_matrix, gradient=True)
            toc = time.time()
            obj_old = f_grad["f"]
            gradient = f_grad["grad"]
            # compute projection of gradient onto tangent space to rotation_matrix
            # projection is zero iff rotation_matrix is a stationary point
            M = np.dot(rotation_matrix.T, gradient)
            S = (M + M.T) / 2
            gradient_proj = gradient - np.dot(rotation_matrix, S)

            # residual
            s = np.linalg.norm(gradient_proj, "fro")
            if (
                s < tol * n_cols
            ) and residual_conv:  # normalize with size of rotation matrix
                dstats["msg"] = "CONVERGED (residual below tol)"
                break

            # Jennings 2001 use the following equivalent expression for the stopping criterion, cf. Eq (8)
            #            v = np.linalg.norm(np.dot((np.eye(n_cols)- np.dot(
            #                    rotation_matrix, rotation_matrix.T)), gradient)) + \
            #                    np.linalg.norm((M - M.T) / 2)
            if not twowaybacktracking:
                lr = 2 * lr
            #            elif i > 0:
            #                alpha = alpha * dstats['s'][-1] / s
            old_d = d
            # backtracking line search for step size alpha
            tic_ls = time.time()
            ls_succeeded = False
            j = 0
            lr_increase = False
            rotation_matrix_candidate = None
            obj_best = obj_old

            # perform Armijo-Goldstein line search with at most 15 steps
            max_steps_ls = 15

            for j in range(1, 1 + max_steps_ls):

                # descent (minimization)
                X = rotation_matrix - lr * gradient_proj

                # compute projection onto manifold of orthogonal matrices
                U, D, V = np.linalg.svd(X)
                rotation_matrix_lr = np.dot(U, V)
                obj_lr = self.get_obj_grad(rotation_matrix_lr, gradient=False)
                if verb >= 3:
                    print("LS%d, obj=%.4f, " % (j, obj_lr["f"]))

                armijocondition = obj_lr["f"] < obj_old - 0.5 * s ** 2 * lr
                
                # s^2 is projected gradient x search direction (=proj. grad)
                if armijocondition:
                    ls_succeeded = True
                    if obj_lr["f"] < obj_best:
                        rotation_matrix_candidate = rotation_matrix_lr
                        obj_best = obj_lr["f"]
                    elif lr_increase:  # TODO: can this be the case?
                        break
                    if j == 1:
                        # if condition is satisfied for the first alpha
                        # try to increase alpha
                        lr_increase = True
                elif lr_increase:
                    # after increasing lr, condition is no longer satisfied
                    lr = lr / 2  # reset to previous lr
                    break

                if armijocondition and (
                        not lr_increase or not twowaybacktracking):
                    break
                lr = 2 * lr if lr_increase else lr / 2
            toc_ls = time.time()
            if verb >= 2:
                info_str = ({1: "INC", 0: "DEC"}[
                    lr_increase] if twowaybacktracking else "")
                print(
                    " | it%d: Line search conducted %d %s steps" %
                    (i, j, info_str))
                print(
                    " | it%d: t(f_grad)=%.2f, t(ls)=%.2f, t(total)=%.2f"
                    % (i, toc - tic, toc_ls - tic_ls, toc_ls - tic)
                )

            if not ls_succeeded and verb >= 0:
                print(
                    "Warning (solve_orthogonal): LS DID NOT CONVERGE (it=%d)" %
                    i)

            if rotation_matrix_candidate is not None:
                rotation_matrix = rotation_matrix_candidate
            else:
                rotation_matrix = rotation_matrix_lr
                obj_best = obj_lr["f"]

            if verb > 1:
                print(rstr % (i, s, obj_best))

            dstats["obj"].append(obj_best)
            dstats["s"].append(s)
            dstats["ls_steps"].append(j)
            dstats["it_times"].append(time.time() - tic)

            obj_diff_rel = abs(obj_old - obj_best) / \
                (1 + max(obj_old, obj_best))
            if objbelowtol_conv and verb >= 2:
                print(" | Rel obj diff=%.6f" % obj_diff_rel)
            # if obj_diff_rel  < 1e-5 and objbelowtol_conv:
            #    dstats['msg'] = "Termination (it=%d): change of obj. val below tol"%(i)
            #    break
        #            d = np.sum(D)
        #            if old_d != 0 and d / old_d < 1 + self.tol:
        #            if old_d != 0 and 1 - self.tol < d / old_d < 1 + self.tol:
        #                if verb >= 0:
        #                    print("Termination (it=%d): change of sing. vals of projection below tol"%(i))
        #                break
        if dstats["msg"] is None:
            dstats["msg"] = "STOPPED (reached max_iter=%d)" % max_iter
        if verb >= 1:
            print(dstats["msg"])
        if self.use_fp:
            self.pool.close()
            self.pool = None
        return self.loadings_rot.copy(), rotation_matrix, dstats

    def _alpha_cf_obj(self, loadings, kappa, alpha=None, gradient=True):
        """
        Crawford-Fergusion family of rotation criteria (to be minimized)
        adds new parameter alpha.
        This code was generated using www.matrixcalculus.org

        alpha: scalar (positive)
            exponent for factor loadings
            alpha=2 yields classical CF family (squared factor loadings)

        kappa: scalar in [0,1]
            convex weight parameter for column complexity
            kappa = 1/n_rows yields CF-Varimax
            kappa = n_cols/(2 n_rows) yields CF-Equamax

        loadings: np.array
            matrix of factor loadings
        """
        if isinstance(alpha, np.ndarray):
            dim = alpha.shape
            assert dim == (1,)
        elif alpha is None:
            return self._cf_obj(loadings, kappa, gradient=gradient)
        if isinstance(kappa, np.ndarray):
            dim = kappa.shape
            assert dim == (1,)
        assert isinstance(loadings, np.ndarray)
        dim = loadings.shape
        assert len(dim) == 2
        loadings_rows = dim[0]
        loadings_cols = dim[1]

        T_0 = np.abs(loadings)
        T_1 = T_0 ** alpha
        t_2 = 1 - kappa
        t_3 = (T_1).dot(np.ones(loadings_cols))
        T_4 = np.outer(t_3, np.ones(loadings_cols)) - T_1

        t_8 = (np.ones(loadings_rows)).dot(T_1)
        T_9 = np.outer(np.ones(loadings_rows), t_8) - T_1
        functionValue = (t_2 * np.sum(((T_1 * T_4)).dot(np.ones(loadings_cols)))) + (
            kappa * np.sum(((T_1 * T_9)).dot(np.ones(loadings_cols)))
        )

        if gradient:
            T_7 = np.sign(loadings)
            t_5 = alpha * t_2
            T_6 = T_0 ** (alpha - 1)
            t_10 = alpha * kappa
            T_11 = T_6 * T_7
            T_12 = (T_1 * T_6) * T_7
            gradient = (
                (
                    (
                        (
                            (t_5 * ((T_4 * T_6) * T_7))
                            + (t_5 * (t_3[:, np.newaxis] * T_11))
                        )
                        - (t_5 * T_12)
                    )
                    + (t_10 * ((T_9 * T_6) * T_7))
                )
                + (t_10 * (T_11 * t_8[np.newaxis, :]))
            ) - (t_10 * T_12)

            return {"f": functionValue, "grad": gradient}
        return {"f": functionValue}

    @staticmethod
    def _cf_obj(loadings, kappa, gradient=True):
        """
        Crawford-Fergusion family of rotation criteria (to be minimized)
        see Browne 2001, Bernaards 2005

        kappa: scalar in [0,1]
            convex weight parameter for column complexity
            kappa = 1/n_rows yields CF-Varimax
            kappa = n_cols/(2 n_rows) yields CF-Equamax

        loadings: np.array
            matrix of factor loadings
        """
        loadings_sq = loadings ** 2

        n, k = loadings.shape

        f = 0
        if gradient:
            grad = np.zeros((n, k))

        if kappa < 1:
            # row complexity
            ones_k = np.ones(k)
            tmp = np.sum(loadings_sq, axis=1)  # sum content of rows
            loadings_sq_n = np.outer(tmp, ones_k)
            #            print(loadings_sq_n)
            loadings_sq_n -= loadings_sq

            f += (1 - kappa) * np.sum(np.multiply(loadings_sq, loadings_sq_n))
            if gradient:
                grad += (1 - kappa) * np.multiply(loadings, loadings_sq_n)
        if kappa > 0:
            # column complexity
            ones_n = np.ones(n)
            #            print(np.sum(loadings_sq, axis=0).shape)
            m_loadings_sq = np.outer(ones_n, np.sum(loadings_sq, axis=0))
            #            print(m_loadings_sq)
            m_loadings_sq -= loadings_sq

            f += kappa * np.sum(np.multiply(loadings_sq, m_loadings_sq))
            if gradient:
                grad += kappa * np.multiply(loadings, m_loadings_sq)

        f = f / 4
        if gradient:
            return {"f": f, "grad": grad}
        return {"f": f}

    def _varimax(self, loadings, init=None, verb=False):
        """
        Perform varimax (orthogonal) rotation, with optional
        Kaiser normalization.

        ** IMPORTANT **
        This uses BSV algorithm from Jennrich (2001) with alpha=0 (see Section 6)

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        rotation_mtx : numpy array, shape (n_factors, n_factors)
            The rotation matrix
        """
        X = loadings.copy()
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        #        if self.normalize:
        #            normalized_mtx = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, X.copy())
        #            X = (X.T / normalized_mtx).T
        #        print("X has shape", X.shape)

        # initialize the rotation matrix
        # to N x N identity matrix#
        if init is None:
            rotation_mtx = np.eye(n_cols)
        else:
            rotation_mtx = init
        d = 0

        obj_grad = self._varimax_obj(np.dot(X, rotation_mtx))
        #        obj2 = obj_grad['criterion']
        grad_full = np.dot(X.T, obj_grad["grad"])
        for _ in range(self.max_iter):

            old_d = d

            loadings = np.dot(X, rotation_mtx)

            # transform data for singular value decomposition)
            #            grad_q = loadings**3 - (1.0 / n_rows) * np.dot(loadings,
            # np.diag(np.sum(loadings ** 2, axis=0)))

            #            print(np.linalg.norm(grad_q-grad_q2))
            #            grad_full = np.dot(X.T, grad_q)

            # perform SVD on the transformed matrix
            U, S, V = np.linalg.svd(grad_full)

            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U, V)
            d = np.sum(S)

            # residual
            #            M = np.dot(rotation_mtx.T, grad_full)
            #            S = (M + M.T) / 2
            #            gradient_proj = grad_full - np.dot(rotation_mtx, S)
            #            print(np.linalg.norm(gradient_proj, 'fro'))

            # report
            #            variances = np.var(np.dot(X, rotation_mtx) ** 2, axis=0)
            #            self.obj = np.sum(variances)
            obj_grad = self._varimax_obj(np.dot(X, rotation_mtx))
            self.obj = obj_grad["f"]
            grad_full = np.dot(X.T, obj_grad["grad"])
            if verb:
                # https://en.wikipedia.org/wiki/Varimax_rotation
                # compute variance of squared loadings
                print("VARIMAX Obj = %.5f" % self.obj)

            # check convergence
            if old_d != 0 and d / old_d < 1 + self.tol:
                break

        # take inner product of loading matrix and rotation matrix
        X = np.dot(X, rotation_mtx)

        # de-normalize the data
        #        if self.normalize:
        #            X = X.T * normalized_mtx
        #        else:
        #            X = X.T

        # convert loadings matrix to data frame
        loadings = X.T.copy()
        return loadings, rotation_mtx

    @np.deprecate
    def _varimax_obj(self, loadings, minimize=False):
        """
        Varimax from Harman (1960) orthomax family, see Jennrich (2001), sec. 6
        """
        loadings_squared = loadings ** 2
        n_rows = loadings.shape[0]
        grad = loadings ** 3 - (1.0 / n_rows) * np.dot(
            loadings, np.diag(np.sum(loadings ** 2, axis=0))
        )

        variances = np.var(loadings_squared, axis=0)
        obj = np.sum(variances)  # TODO: does not match grad
        if minimize:
            obj *= -1
            grad *= -1
        return {"f": obj, "grad": grad}

def get_classprobs(mean_logits, loadings, gradient, onesided=False):
    """
    compute flow probabilities for a pixel

    mean_logits: np.array
        mean logits for the classes of the pixel

    loadings: np.array
        vector of slopes for the classes (factor loadings)

    Returns
    ------------
    class_probs: np.array
        class probababilities (expected values under standard normal
        distribution of the factor variable)
    """
    num_c = mean_logits.shape[-1]
    npixels = mean_logits.shape[0]
    class_probs = np.zeros_like(mean_logits)  # TODO: could allocate externally
    if gradient:
        grad = np.zeros((npixels, num_c, num_c))

    d_ipts = []  # TODO: use numpy array

    # dummy vector for "intersection point" of parallel class slopes
    hundreds = np.ones(npixels) * 100

    for i in range(num_c):  # choose 2 out of num_c classes
        for j in range(i):
            delta_slope = loadings[:, i] - loadings[:, j]
            # is zero if loadings are equal in a pixel for class i and j
            # NaN values occur for parallel lines

            # avoid NaNs by replacing inter_pt of parallel lines by large
            # number
            inter_pt = np.where(
                delta_slope != 0,
                -(mean_logits[:, i] - mean_logits[:, j]) / delta_slope,
                hundreds,
            )
            d_ipts.append(inter_pt)

    sorted_id = np.argsort(d_ipts, 0)
    ipts = np.sort(d_ipts, 0)

    # old_z = ipts[:,0] - 10 # -10 gives value to the left of smallest ipt
    # -100 is far out in the tail (~zero prob to the left)
    z_l_interval = -100  # keep track of lower end of interval for current class
    z_l_class = np.ones(npixels) * (
        -100
    )  # keep track of lower end of interval for current class

    z_underline = np.ones((npixels, num_c)) * (-100)
    z_overline = np.ones((npixels, num_c)) * (-100)

    # determine predicted class on each interval
    cl = None
    idx_list = np.arange(sorted_id.shape[-1])

    # add upper bound to intersetction points --> last intersection +10
    ipts = np.concatenate([ipts, ipts[-1:, idx_list] + 10], axis=0)

    for zi in ipts:
        # compute midpoint of interval
        z = np.expand_dims((z_l_interval + zi) / 2, -1)

        # get prediction for current linear segment
        cl_interval = np.argmax(mean_logits + z * loadings, axis=-1)

        # is it first iteration --> no class derived yet --> use class from
        # first interval
        if cl is None:
            cl = cl_interval
        else:
            # Only work pixels where the class actually changed
            # get indices, where class changed
            idx_sub_list = idx_list[cl != cl_interval]

            # intervals where the classes changed
            cl_interval_sub = cl_interval[idx_sub_list]

            # list of classes that need to be updated
            cl_sub = cl[idx_sub_list]

            # get probability for currently considered interval
            #            class_probs[idx_sub_list, cl_sub] = (
            #                    ndtr(z_l_interval[idx_sub_list]) -
            #                    ndtr(z_l_class[idx_sub_list]))

            z_underline[idx_sub_list, cl_sub] = z_l_class[idx_sub_list]
            z_overline[idx_sub_list, cl_sub] = z_l_interval[idx_sub_list]

            if gradient:
                delta_slope = (
                    loadings[idx_sub_list, cl_interval_sub]
                    - loadings[idx_sub_list, cl_sub]
                )
                delta_mean = (
                    mean_logits[idx_sub_list, cl_sub]
                    - mean_logits[idx_sub_list, cl_interval_sub]
                )

                # if logit functions are parallel, the intersection points are
                # x=-infty and x=infty --> x**2 = infty
                # for delta_slope --> 0: exp(-x**2) converges fast to than 1/delta_slope**2
                # hence set gradient to zero
                x = delta_mean / delta_slope

                gradval = np.where(
                    delta_slope != 0,
                    (delta_mean / delta_slope ** 2)
                    * np.exp(-(x ** 2) / 2)
                    / np.sqrt(2 * np.pi),
                    np.zeros_like(delta_mean),
                )

                gradval = np.transpose(gradval)

                # check if NaN exclusion works --> is the approach above
                # numerical stable or do we need some eps?
                is_not_nan = np.logical_not(np.isnan(x))
                assert np.all(is_not_nan)

                # update gradients for affected indices
                grad[idx_sub_list, cl_sub, cl_sub] += gradval
                grad[idx_sub_list, cl_sub, cl_interval_sub] -= gradval
                grad[idx_sub_list, cl_interval_sub, cl_interval_sub] += gradval
                grad[idx_sub_list, cl_interval_sub, cl_sub] -= gradval

            z_l_class[idx_sub_list] = z_l_interval[idx_sub_list]

            # update current classes
            cl[idx_sub_list] = cl_interval[idx_sub_list]

        z_l_interval = zi
    z_underline[idx_list, cl] = z_l_class
    z_overline[idx_list, cl] = 100
    class_probs = ndtr(z_overline) - ndtr(z_underline)
    #    class_probs[idx_list, cl] = 1 - ndtr(z_l_class)
    out_dict = {}
    out_dict["f"] = class_probs

    if onesided:
        out_dict["fpos"] = ndtr(np.maximum(z_overline, 0)) - ndtr(
            np.maximum(z_underline, 0)
        )
        out_dict["fneg"] = ndtr(np.minimum(z_overline, 0)) - ndtr(
            np.minimum(z_underline, 0)
        )

    if gradient:
        out_dict["grad"] = grad

    return out_dict

def get_flowprobs(
        mean_logits,
        loadings,
        num_c,
        gradient=True,
        pool=None,
        usepool=True,
        onesided=False):
    """
    Calculate flow probabibilities (one vs. all others)

    Parameters
    ----------
    mean_logits   :   nparray   Mean logits of the factor model.
    loadings      :   nparray   Loading matrix of the factor model.
    num_c         :   int       Number of classes in underlying seg. problem.
    gradient      :   bool      Also compute and return the gradient. 
    pool          :   Pool      Pool to realize parallel computations.
    usepool       :   bool      Use pool for parallel computation on cpus.
    onesided      :   bool      Return one-sided flow probabilities.
    """

    try:
        mean_logits = rearrange(mean_logits, "(p c) -> p c", c=num_c)
        loadings = rearrange(loadings, "(p c) k -> p c k", c=num_c)
    except BaseException:
        assert mean_logits.shape[1] == num_c
        assert loadings.shape[1] == num_c
    npixels, num_c, rank = loadings.shape

    if pool is None and usepool:
        pool_loc = mp.Pool(mp.cpu_count())
    elif not usepool:
        pool_loc = itertools
    else:
        pool_loc = pool  # create and close pool outside of function
    # do not run this from IPython console --
    # https://stackoverflow.com/questions/34086112/python-multiprocessing-pool-stuck
    arglist = [[mean_logits, loadings[:, :, j], gradient, onesided]
               for j in range(rank)]
    result = pool_loc.starmap(get_classprobs, arglist)
    if pool is None and usepool:
        pool_loc.close()
    #    result = starmap(get_classprobs, [[mean_logits, loadings[:,:,j], gradient] for j in range(rank)])

    out_dict = {}
    fchars = ["f"]
    if onesided:
        fchars += ["fpos", "fneg"]
    if gradient:
        grad = np.empty((npixels, rank, num_c, num_c))
    for fchar in fchars:  # initialize flow probabilities
        out_dict[fchar] = np.zeros((npixels, num_c, rank))
    for j, class_probs in enumerate(result):
        if gradient:
            grad[:, j, :, :] = class_probs["grad"]
        for fchar in fchars:
            fps = out_dict[fchar]
            fps[:, :, j] = class_probs[fchar]
            fps[np.arange(len(fps)), np.argmax(mean_logits, axis=-1), j] -= 1
            # last line does not affect grad

    for fchar in fchars:
        out_dict[fchar] = rearrange(out_dict[fchar], "p c k -> (p c) k")
    if gradient:
        out_dict["grad"] = grad
    return out_dict
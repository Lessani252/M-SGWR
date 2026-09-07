from __future__ import annotations
import spreg.user_output as USER
from .diagnostics import get_AIC, get_AICc, get_BIC
from .summary import summaryModel, summaryGLM, summaryMSGWR
import copy
from typing import Optional
import numpy as np
import numpy.linalg as la
from scipy.stats import t
from scipy.special import factorial
from itertools import combinations as combo
from spglm.family import Gaussian, Poisson, Binomial
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls, _compute_betas_gwr
from spglm.utils import cache_readonly
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

import math

import numpy as np
from scipy.stats import rankdata         

class SGWR(GLM):
        
    def __init__(self, coords, y, X, data, bw, bt_value, att_bw, family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel=None, fixed=False, constant=True,
                 spherical=False, hat_matrix=False, n_jobs=False):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel or ('gaussian' if fixed else 'bisquare')
        expected_kernel = 'gaussian' if fixed else 'bisquare'
        if self.kernel.lower() != expected_kernel:
            raise NotImplementedError('MSGWR supports fixed gaussian or adaptive bisquare kernels.')
        self.kernel = self.kernel.lower()
        if spherical:
            raise NotImplementedError('MSGWR currently requires projected coordinates.')
        if not isinstance(family, Gaussian):
            raise NotImplementedError('MSGWR currently supports Gaussian responses only.')
        self.fixed = fixed
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}

        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.spherical = spherical
        self.hat_matrix = hat_matrix
        self.n_jobs = n_jobs
        
        self.bt_value = bt_value
        self.data = data
        self.att_bw = att_bw
        
    def _build_wi(self, i, bw, bt_value=None, att_bw=None, data=None):
        from .weights import composite_weights
        return composite_weights(
            self.coords, i, bw,
            self.bt_value if bt_value is None else bt_value,
            self.data if data is None else data, self.fixed)

    def _local_fit(self, i):
        
        wi = self._build_wi(i, self.bw, self.bt_value, self.att_bw, self.data).reshape(-1, 1)  

        if isinstance(self.family, Gaussian):
            betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
            predy = np.dot(self.X[i], betas)[0]
            resid = self.y[i] - predy
            influ = np.dot(self.X[i], inv_xtx_xt[:, i])
            w = 1

        if self.fit_params['lite']:
            
            return influ, resid, predy, betas.reshape(-1)
        else:
            Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si * w * w)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
            if not self.hat_matrix:
                Si = None
            
            return influ, resid, predy, betas.reshape(-1), w, Si, tr_STS_i, CCT

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls',
            lite=False, pool=None):

        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite

        if solve.lower() == 'iwls':

            if self.points is None:
                m = self.y.shape[0]
            else:
                m = self.points.shape[0]

            if pool:
                rslt = pool.map(self._local_fit, range(m))  #parallel using mp.Pool
            else:
                rslt = map(self._local_fit, range(m))  #sequential

            rslt_list = list(zip(*rslt))
            influ = np.array(rslt_list[0]).reshape(-1, 1)
            resid = np.array(rslt_list[1]).reshape(-1, 1)
            params = np.array(rslt_list[3])

            if lite:
                return SGWRResultsLite(self, resid, influ, params)
            predy = np.asarray(rslt_list[2]).reshape(-1, 1)
            w = np.asarray(rslt_list[4]).reshape(-1, 1)
            S = np.asarray(rslt_list[5]) if self.hat_matrix else None
            tr_STS = np.sum(rslt_list[6])
            CCT = np.asarray(rslt_list[7])
            return SGWRResults(self, params, predy, S, CCT, influ, tr_STS, w)

class SGWRResults(GLMResults):

    def __init__(self, model, params, predy, S, CCT, influ, tr_STS=None,
                 w=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.tr_STS = tr_STS
        self.influ = influ
        self.CCT = self.cov_params(CCT, model.exog_scale)
        self._cache = {}

    @cache_readonly
    def resid_ss(self):
        if self.model.points is not None:
            raise NotImplementedError('Not available for M-SGWR prediction')
        else:
            u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self, scale=None):
        if isinstance(self.family, Gaussian):
            scale = self.sigma2
        else:
            scale = 1.0
        return scale

    def cov_params(self, cov, exog_scale=None):

        if exog_scale is not None:
            return cov * exog_scale
        else:
            return cov * self.scale

    @cache_readonly
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        return np.sum(self.influ)

    @cache_readonly
    def ENP(self):

        if self.model.sigma2_v1:
            return self.tr_S
        else:
            return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def y_bar(self):
        """
        weighted mean of y
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):
            w_i = np.reshape(self.model._build_wi(i, self.model.bw), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar

    @cache_readonly
    def TSS(self):

        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * (self.y.reshape(
                               (-1, 1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):

        if self.model.points is not None:
            n = len(self.model.points)
            resid = self.model.exog_resid.reshape((-1, 1))
        else:
            n = self.n
            resid = self.resid_response.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))
        for i in range(n):
            RSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * resid**2)
        return RSS

    @cache_readonly
    def sigma2(self):
        if self.model.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
            # could be changed to SWSTW - nothing to test against
            return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def std_res(self):

        return self.resid_response.reshape(
            (-1, 1)) / (np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):

        return np.sqrt(self.CCT)

    @cache_readonly
    def cooksD(self):

        return self.std_res**2 * self.influ / (self.tr_S * (1.0 - self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1, 1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(
                2.0 * self.W * (y * np.log(y / (ybar * off)) -
                                (y - ybar * off)), axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1, 1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1, 1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(), self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1, 1))

    @cache_readonly
    def pDev(self):

        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance / self.deviance)

    @cache_readonly
    def adj_alpha(self):

        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):

        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):

        n = self.n
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0 * self.tr_S + self.tr_STS

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)

    @cache_readonly
    def global_deviance(self):
        deviance = np.sum(self.family.resid_dev(self.y, self.mu)**2)
        return deviance

    @cache_readonly
    def D2(self):
        """
        Percentage of deviance explanied. Equivalent to 1 - (deviance/null deviance)
        """
        D2 = 1.0 - (self.global_deviance / self.null_deviance)
        return D2

    @cache_readonly
    def R2(self):
        """
        Global r-squared value for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.D2
        else:
            raise NotImplementedError('R2 only for Gaussian')

    @cache_readonly
    def adj_D2(self):
        """
        Adjusted percentage of deviance explanied.
        """
        adj_D2 = 1 - (1 - self.D2) * (self.n - 1) / (self.n - self.ENP - 1)
        return adj_D2

    @cache_readonly
    def adj_R2(self):
        """
        Adjusted global r-squared for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.adj_D2
        else:
            raise NotImplementedError('adjusted R2 only for Gaussian')

    @cache_readonly
    def aic(self):
        return get_AIC(self)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def bic(self):
        return get_BIC(self)


class SGWRResultsLite(object):
    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.family = model.family
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.model = model
        self.params = params

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y.ravel(), self.mu.ravel())

    @cache_readonly
    def mu(self):
        return self.y - self.resid_response

    @cache_readonly
    def predy(self):
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)

class MSGWR(SGWR):        
        
    def __init__(self, coords, y, X, data, selector, sigma2_v1=True,
                 kernel=None, fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
        self.selector = selector
        self.bws = selector.bw[0]  #final set of bandwidth
        self.bws_history = selector.bw[1]  #bws history in backfitting
        self.alpha_history = selector.bw[-2] # alpha history
        self.opt_alpha_vec = np.array([selector.bw[-1][j][-1] for j in range(len(self.bws))]) ## list of final optimal alphas
        self.bw_init = selector.bw_init  #initialization bandiwdth

        att_bw = 50 ### naser added for attribute bw as initial value
        self.family = Gaussian()  # manually set since we only support Gassian MGWR for now
        bt_value = 1 
        SGWR.__init__(self, coords, y, X, data, self.bw_init, bt_value, att_bw, family=self.family,
                     sigma2_v1=sigma2_v1, kernel=kernel, fixed=fixed,
                     constant=constant, spherical=spherical,
                     hat_matrix=hat_matrix)
        self.selector = selector
        self.sigma2_v1 = sigma2_v1
        self.points = None
        self.P = None
        self.exog_resid = None
        self.exog_scale = None
        self.fit_params = {}
        
        self.bt_value = bt_value ## we just put this here and within the model the model extracts the necessary bt_value from the selector history like bandwidth
                            ## for final model we can design such that remove this part from 'mgwr_res' call
        if self.constant:
            self.data, keep_data,warn = USER.check_constant(data)
        self.data = np.asarray(self.data, dtype=float)
        if self.data.shape != self.X.shape:
            raise ValueError('data must have one attribute column per model covariate.')
        if (selector.fixed != self.fixed or selector.constant != self.constant
                or selector.kernel != self.kernel or selector.spherical != self.spherical):
            raise ValueError('MSGWR settings must match the fitted selector.')
        selector_X = USER.check_constant(selector.X_loc)[0] if self.constant else selector.X_loc
        selector_data = USER.check_constant(selector.data)[0] if self.constant else selector.data
        if not (np.array_equal(self.coords, selector.coords)
                and np.array_equal(self.y, selector.y)
                and np.array_equal(self.X, selector_X)
                and np.array_equal(self.data, selector_data)):
            raise ValueError('MSGWR inputs must match the fitted selector.')
    
    def _alpha_his_to_array(self, alpha_hist_dict, n_iter, k):
        alpha_matrix = np.full((n_iter, k), np.nan)  
        for j in range(k):
            if j in alpha_hist_dict:
                for iter_i, (bw, alpha, _) in enumerate(alpha_hist_dict[j]):
                    alpha_matrix[iter_i, j] = alpha
        return alpha_matrix
    
    def _attbw_his_to_array(self, alpha_hist_dict, n_iter, k):
        att_bw_matrix = np.full((n_iter, k), np.nan)
        for j in range(k):
            if j in alpha_hist_dict:
                for iter_i, (_, _, att_bw) in enumerate(alpha_hist_dict[j]):
                    att_bw_matrix[iter_i, j] = att_bw
        return att_bw_matrix


    def _chunk_compute_R(self, chunk_id=0):
        """Replay selection's linear smoothers, conditional on selected weights."""
        n, k = self.n, self.k
        indices = np.array_split(np.arange(n), self.n_chunks)[chunk_id]
        identity = np.zeros((n, len(indices)))
        identity[indices, np.arange(len(indices))] = 1
        pR = np.zeros((n, len(indices), k))
        coefficient_maps = np.zeros_like(pR)
        # Selection starts with a joint, purely spatial SGWR fit.
        for i in range(n):
            wi = self._build_wi(i, self.bw_init, 1.0).reshape(-1)
            xw = self.X.T * wi
            coefficient_maps[i] = np.linalg.solve(xw @ self.X, xw)[:, indices].T
            pR[i] = coefficient_maps[i] * self.X[i]
        err = identity - pR.sum(axis=2)
        for iteration, bandwidths in enumerate(self.bws_history):
            for j in range(k):
                partial = pR[:, :, j] + err
                alpha = self.selector.bw[-2][j][iteration][1]
                xj = self.X[:, j]
                for i in range(n):
                    wi = self._build_wi(i, bandwidths[j], alpha, data=self.data[:, j])
                    xw = xj * wi
                    coefficient_maps[i, :, j] = (xw @ partial) / (xw @ xj)
                pR[:, :, j] = coefficient_maps[:, :, j] * xj[:, None]
                err = partial - pR[:, :, j]
        CCT = np.sum(coefficient_maps ** 2, axis=1)
        ENP_j = np.sum(pR[indices, np.arange(len(indices)), :], axis=0)
        S_chunk = pR.sum(axis=2)
        tr_STS = np.sum(S_chunk ** 2)
        influence = np.zeros(n)
        influence[indices] = S_chunk[indices, np.arange(len(indices))]
        return ENP_j, CCT, pR if self.hat_matrix else None, tr_STS, influence

    def fit(self, n_chunks=1, pool=None):
        if not isinstance(n_chunks, (int, np.integer)) or n_chunks < 1:
            raise ValueError('n_chunks must be a positive integer.')
        params = self.selector.params
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        try:
            from tqdm.autonotebook import tqdm  
        except ImportError:

            def tqdm(x, total=0,
                     desc=''): 
                return x

        if pool:
            self.n_chunks = pool._processes * n_chunks
            rslt = tqdm(
                pool.imap(self._chunk_compute_R, range(self.n_chunks)),
                total=self.n_chunks, desc='Inference')

        else:
            self.n_chunks = n_chunks
            rslt = map(self._chunk_compute_R, tqdm(range(self.n_chunks), desc='Inference'))
            # rslt = map(self._chunk_compute_R, range(self.n_chunks))

        rslt_list = list(zip(*rslt))
        ENP_j = np.sum(np.array(rslt_list[0]), axis=0)
        CCT = np.sum(np.array(rslt_list[1]), axis=0)

        w = np.ones(self.n)
        if self.hat_matrix:
            R = np.hstack(rslt_list[2])
        else:
            R = None
            
        tr_STS = np.sum(rslt_list[3])
        influence = np.sum(rslt_list[4], axis=0).reshape(-1, 1)
        return MSGWRResults(self, params, predy, CCT, ENP_j, w, R, tr_STS, influence)

    def exact_fit(self):
        """Solve the converged backfitting equations at the final weights.

        This can differ from fit() when selection stops before convergence.
        It requires O((n*k)**2) memory.
        """
        smoothers, coefficient_smoothers = [], []
        for j in range(self.k):
            xj = self.X[:, j]
            weights = np.array([self._build_wi(i, self.bws[j], self.opt_alpha_vec[j],
                                data=self.data[:, j]) for i in range(self.n)])
            xw = weights * xj
            B = xw / (xw @ xj)[:, None]
            coefficient_smoothers.append(B)
            smoothers.append(xj[:, None] * B)
        identity = np.eye(self.n)
        system = np.block([[identity if j == h else smoothers[j]
                            for h in range(self.k)] for j in range(self.k)])
        stacked_R = np.linalg.solve(system, np.vstack(smoothers))
        R = np.stack(np.split(stacked_R, self.k), axis=2)
        S = R.sum(axis=2)
        maps = [coefficient_smoothers[j] @ (identity - S + R[:, :, j])
                for j in range(self.k)]
        params = np.column_stack([(B @ self.y).ravel() for B in maps])
        CCT = np.column_stack([np.sum(B**2, axis=1) for B in maps])
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)
        return MSGWRResults(self, params, predy, CCT,
                            np.trace(R, axis1=0, axis2=1), np.ones(self.n),
                            R if self.hat_matrix else None,
                            np.sum(S**2), np.diag(S).reshape(-1, 1), inference_method='exact')

    def predict(self):
        '''
        Not implemented.
        '''
        raise NotImplementedError('N/A')


class MSGWRResults(SGWRResults):
    def __init__(self, model, params, predy, CCT, ENP_j, w, R, tr_STS, influ, inference_method='history'):
        """
        Initialize class
        """
        self.inference_method = inference_method
        self.ENP_j = ENP_j
        self.R = R
        SGWRResults.__init__(self, model, params, predy, None, CCT, influ, tr_STS=tr_STS, w=w)
        if model.hat_matrix:
            self.S = np.sum(self.R, axis=2)
        self.predy = predy

    @cache_readonly
    def tr_S(self):
        return np.sum(self.ENP_j)

    @cache_readonly
    def W(self):
        Ws = []
        count =0  ### Naser 
        for bw_j in self.model.bws:
            opt_alpha = self.model.opt_alpha_vec[count] 
            data = self.model.data[:, count]
            att_bw = 100 
            W = np.array(
                [self.model._build_wi(i, bw_j, opt_alpha, att_bw, data) for i in range(self.n)])
            
            if W.shape[-1] == 1:
                W = W.squeeze(-1)  
            else:
                W = W

            Ws.append(W)
            count +=1
            
        return Ws

    @cache_readonly
    def adj_alpha_j(self):
        alpha = np.array([.1, .05, .001])
        pe = np.array(self.ENP_j).reshape((-1, 1))
        p = 1.
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    #Function for getting BWs intervals
    def get_bws_intervals(self, selector, level=0.95):

        intervals = []
        try:
            import pandas as pd
        except ImportError:
            return

        for j in range(self.k):
            aiccs = pd.DataFrame(list(zip(*selector.sel_hist[-self.k+j]))[1],columns=["aicc"])
            aiccs['bw'] = list(zip(*selector.sel_hist[-self.k+j]))[0]
            aiccs = aiccs.sort_values(by=['aicc'])
            d_aic_ak = aiccs.aicc - aiccs.aicc.min()
            w_aic_ak = np.exp(-0.5*d_aic_ak) / np.sum(np.exp(-0.5*d_aic_ak))
            aiccs['w_aic_ak'] = w_aic_ak/np.sum(w_aic_ak)
            aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
            index = len(aiccs[aiccs.cum_w_ak < level]) + 1
            interval = (aiccs.iloc[:index,:].bw.min(),aiccs.iloc[:index,:].bw.max())
            intervals += [interval]
        return intervals


    def local_collinearity(self):
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = self.n
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))
        local_VIF = np.full((nrow, nvar), np.nan) ## 3/14/2026
        ridge=1e-12 
        eps=1e-12
        
        for i in range(nrow):
            xw = np.zeros((x.shape))
            for j in range(nvar):
                wi = w[j][i]
                sw = np.sum(wi)
                wi = wi / sw
                xw[:, j] = x[:, j] * wi

            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]

            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

            # -----------------------------
            # Local VIF
            # -----------------------------
            for j in range(nvar):
    
                # skip intercept if present
                if self.model.constant and j == 0:
                    continue
    
                wi = w[j][i]
                sw = np.sum(wi)
                if sw <= eps:
                    continue
    
                wi = wi / sw
    
                y_aux = x[:, j]
                X_others = np.delete(x, j, axis=1)
    
                sw_sqrt = np.sqrt(wi).reshape(-1, 1)
                Xw = X_others * sw_sqrt
                yw = y_aux.reshape(-1, 1) * sw_sqrt
    
                XtX = Xw.T @ Xw
                XtX += ridge * np.eye(XtX.shape[0])
    
                Xty = Xw.T @ yw
    
                try:
                    beta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    continue
    
                yhat = (X_others @ beta).reshape(-1)
    
                ybar = np.sum(wi * y_aux)
                sst = np.sum(wi * (y_aux - ybar)**2)
                sse = np.sum(wi * (y_aux - yhat)**2)
    
                if sst > eps:
                    r2 = 1 - sse / sst
                    local_VIF[i, j] = 1 / max(eps, (1 - r2))

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]
        
        return local_CN, VDP, local_VIF

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        temp_sel = copy.deepcopy(selector)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        search_params = temp_sel.search_params

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        try:
            from tqdm.auto import tqdm  
        except ImportError:

            def tqdm(x, desc=''):  
                return x

        for x in tqdm(range(n_iters), desc='Testing'):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_sel.search(**search_params)
            temp_params = temp_sel.params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    def summary(self, as_str: bool = True) -> Optional[str]:
        
        """Return the report; use as_str=False to print it instead."""
        report = summaryModel(self) + summaryGLM(self) + summaryMSGWR(self)
        if as_str:
            return report
        print(report)
        return None

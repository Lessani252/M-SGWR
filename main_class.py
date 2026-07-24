"""
Originally this code was for mgwr model (Taylor Oshan tayoshan@gmail.com) and SGWR (M. Naser Lessani). However, the code has been modefied according to M-SGWR model.
"""
from __future__ import annotations

__author__ = "M. Naser Lessani naserlessani252@gmail.com"

import copy
from typing import Optional
import copy
import math
import multiprocessing as mp
from itertools import combinations as combo
from typing import Literal, Optional

import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import cdist
from scipy.special import factorial
from scipy.stats import rankdata, t
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler
from spglm.family import Gaussian
from spglm.glm import GLM, GLMResults
from spglm.iwls import _compute_betas_gwr, iwls
from spglm.utils import cache_readonly

from Diagnostics import get_AIC, get_AICc, get_BIC

class SGWR(GLM):
        
    def __init__(self, coords, y, X, bw, bt_value, data, att_bw, variable, family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False, n_jobs=False):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel
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
        self.variables = variable
        self.att_bw = att_bw
        
    def _build_wi(self, i, bw, bt_value, att_bw, data): 
        if bt_value < 1 and not np.all(data == 1):
            dist = cdist([self.coords[i]], self.coords).reshape(-1)
            if data.ndim == 1:
                if self.fixed:
                    wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)

                else:
                    maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                    zs = dist / maxd
                    zs[zs >= 1] = 1
                    wg = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)

                data = data.flatten().reshape(-1, 1)
                
                neighbor_indices = np.where(wg.flatten() > 0)[0]
                # neighbor_indices = np.where(wg.flatten() >= 0)[0]
                X_neighbors = data[neighbor_indices]
                SD = np.std(X_neighbors)
                xi = data[i]  
                diff_squared = ((X_neighbors - xi) / SD) ** 2
                ws_neighbors = np.exp(np.log(0.5) * diff_squared)  
                ws= np.zeros_like(data)
                ws[neighbor_indices] = ws_neighbors

                wi = bt_value * wg + (1 - bt_value) * ws

                wi[wi>=1] = 1
            
                return wi

            else:
                dist = cdist([self.coords[i]], self.coords).reshape(-1)
                if self.fixed:
                    wg = np.exp(-0.5 * (dist / bw) ** 2).flatten()

                else:
                    maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                    zs = dist / maxd
                    zs[zs >= 1] = 1
                    wg = ((1 - (zs) ** 2) ** 2).flatten()
                    
                data = data.flatten()
                neighbor_indices = np.where(wg.flatten() > 0)[0]
                X_neighbors = data[neighbor_indices]
                SD = np.std(X_neighbors)
                xi = data[i]  
                diff_squared = ((X_neighbors - xi) / SD) ** 2
                ws_neighbors = np.exp(np.log(0.5) * diff_squared)
                
                ws= np.zeros_like(data)
                ws[neighbor_indices] = ws_neighbors
                
                wi = bt_value * wg + (1 - bt_value) * ws 

                wi[wi>=1] = 1

                return wi

        else: 
            dist = cdist([self.coords[i]], self.coords).reshape(-1)

            if self.fixed: ## this is fixed gaussian
                wi = np.exp(-0.5 * (dist / bw) ** 2).flatten()

            else:

                maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                zs = dist / maxd
                zs[zs >= 1] = 1
                wi = ((1 - (zs) ** 2) ** 2).flatten()

            return wi
                

    def _local_fit(self, i):
        
        wi = self._build_wi(i, self.bw, self.bt_value, self.att_bw, self.X).reshape(-1, 1)  

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

            return SGWRResultsLite(self, resid, influ, params)

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
        return self.family.loglike(self.y, self.mu)

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
        
    def __init__(self, coords, y, X, selector, data, variable, sigma2_v1=True,
                 kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
        self.selector = selector
        self.bws = self.selector.bw[0]  
        self.bws_history = selector.bw[1]  
        self.alpha_history = selector.bw[-2] 
        self.opt_alpha_vec = np.array([v[-1] for v in selector.bw[-1].values()]) 
        self.bw_init = self.selector.bw_init  

        att_bw = 50 
        self.family = Gaussian()  
        bt_value = 1 
        SGWR.__init__(self, coords, y, X, self.bw_init, bt_value, data, att_bw, variable, family=self.family,
                     sigma2_v1=sigma2_v1, kernel=kernel, fixed=fixed,
                     constant=constant, spherical=spherical,
                     hat_matrix=hat_matrix)
        self.selector = selector
        self.sigma2_v1 = sigma2_v1
        self.points = None
        self.P = None
        self.offset = None
        self.exog_resid = None
        self.exog_scale = None
        self_fit_params = None
        
        self.bt_value = bt_value 
        self.data = data
        self.variables = variable
    
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
        """
        Compute MGWR inference by chunks to reduce memory footprint.
        """
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))
        
        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) *chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),k))  #partial R: n by chunk_size by k

        err = init_pR - np.sum(pR, axis=2)  #n by chunk_size
        
        n_iter = self.bws_history.shape[0]
        self.alpha_history = self._alpha_his_to_array(self.selector.bw[-2], n_iter, k)
        self.attbw_history = self._attbw_his_to_array(self.selector.bw[-2], n_iter, k)
        weights = 0

        if np.allclose(np.asarray(self.alpha_history[-1]), 1.0): 
            for iter_i in range(self.bws_history.shape[0]): 
                for j in range(k):
                    pRj_old = pR[:, :, j] + err
        
                    Xj = self.X[:, j]
        
                    n_chunks_Aj = n_chunks
        
                    chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                    for chunk_Aj in range(n_chunks_Aj):
        
                        chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(chunk_Aj + 1) * chunk_size_Aj]
        
                        pAj = np.empty((len(chunk_index_Aj), n))
                        for i in range(len(chunk_index_Aj)):
                            index = chunk_index_Aj[i]
                            alpha = self.alpha_history[-1][j]
                            wi = self._build_wi(index, self.bws_history[-1][j], alpha, j, Xj)
                            wi = wi.reshape(-1)
                            ###
                            wi = wi.flatten() 
                            xw = Xj * wi
                            pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
        
                        pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                    err = pRj_old - pR[:, :, j]
        else:
            for j in range(k):
                pRj_old = pR[:, :, j] + err
    
                Xj = self.X[:, j]
    
                n_chunks_Aj = n_chunks
    
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
    
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(chunk_Aj + 1) * chunk_size_Aj]
    
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        
                        alpha = self.alpha_history[-1][j]
                        wi = self._build_wi(index, self.bws_history[-1][j], alpha, j, Xj)
                        wi = wi.reshape(-1)
                        ###
                        wi = wi.flatten() 
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
    
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]
            
        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        if self.hat_matrix:
            return ENP_j, CCT, pR
        return ENP_j, CCT

    def fit(self, n_chunks=1, pool=None):
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

        rslt_list = list(zip(*rslt))
        ENP_j = np.sum(np.array(rslt_list[0]), axis=0)
        CCT = np.sum(np.array(rslt_list[1]), axis=0)

        w = np.ones(self.n)
        if self.hat_matrix:
            R = np.hstack(rslt_list[2])
        else:
            R = None
            
        return MSGWRResults(self, params, predy, CCT, ENP_j, w, R)


    def exact_fit(self):

        P = []
        Q = []
        I = np.eye(self.n)
        for j1 in range(self.k):
            Aj = SGWR(self.coords,self.y,self.X[:,j1].reshape(-1,1),bw=self.bws[j1],hat_matrix=True,constant=False).fit().S
            Pj = []
            for j2 in range(self.k):
                if j1 == j2:
                    Pj.append(I)
                else:
                    Pj.append(Aj)
            P.append(Pj)
            Q.append([Aj])

        P = np.block(P)
        Q = np.block(Q)
        R = np.linalg.solve(P, Q)
        f = R.dot(self.y)

        params =  f/self.X.T.reshape(-1,1)
        params = params.reshape(-1,self.n).T

        R = np.stack(np.split(R,self.k),axis=2)
        ENP_j = np.trace(R, axis1=0, axis2=1)
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)
        w = np.ones(self.n)

        CCT = np.zeros((self.n,self.k))
        for j in range(self.k):
            CCT[:, j] = ((R[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(axis=1)

        return MSGWRResults(self, params, predy, CCT, ENP_j, w, R)


    def predict(self):
        '''
        Not implemented.
        '''
        raise NotImplementedError('N/A')


class MSGWRResults(SGWRResults):
    def __init__(self, model, params, predy, CCT, ENP_j, w, R):
        """
        Initialize class
        """
        self.ENP_j = ENP_j
        self.R = R
        SGWRResults.__init__(self, model, params, predy, None, CCT, None, w)
        if model.hat_matrix:
            self.S = np.sum(self.R, axis=2)
        self.predy = predy

    @cache_readonly
    def tr_S(self):
        return np.sum(self.ENP_j)

    @cache_readonly
    def W(self):
        Ws = []
        count =0  
        for bw_j in self.model.bws:
            opt_alpha = self.model.opt_alpha_vec[count] 
            data = self.X[:, count]
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

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return local_CN, VDP

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
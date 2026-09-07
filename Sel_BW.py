import spreg.user_output as USER
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV
from .msgwr import SGWR
from .alpha import golden_section, equal_interval, multi_bw
import warnings
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from spglm.family import Gaussian

getDiag = {'AICc': get_AICc, 'AIC': get_AIC, 'BIC': get_BIC, 'CV': get_CV}


class Sel_BW(object):
    def __init__(self, coords, y, X_loc, data, X_glob=None, parameter=None, family=Gaussian(),
                 offset=None, kernel=None, fixed=False, multi=False, alphacurve=False, pmutation=False, 
                 constant=True, spherical=False, n_jobs=-1):
        self.coords = np.array(coords)
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family = family
        self.fixed = fixed
        self.kernel = (kernel or ('gaussian' if fixed else 'bisquare')).lower()
        
        
        self.bt_value = 1 ## just an initial value
        self.data = data
        self.alphacurve = alphacurve
        self.att_bw = 50 # this value doens't influence the first initialization step since alpha is (1) thus it can be random
        
        if offset is None:
            self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self.n_jobs = n_jobs
        self.search_params = {}
        self.parameter = parameter
        self.pmutation = pmutation
        
    def search(self, search_method='golden_section', criterion='AICc',
               bw_min=None, bw_max=None, interval=0.0, tol=1.0e-6,
               max_iter=200, init_multi=None, tol_multi=1.0e-5,
               rss_score=False, max_iter_multi=200, multi_bw_min=[None],
               multi_bw_max=[None
                             ], bws_same_times=5, verbose=False,pool=None):
        
        if max_iter < 1 or max_iter_multi < 1:
            raise ValueError('Iteration limits must be positive.')
        if criterion not in getDiag:
            raise ValueError('Unknown selection criterion.')
        if self.multi and criterion not in ('AICc', 'CV'):
            raise NotImplementedError('Multiscale alpha selection supports AICc or CV.')
        k = USER.check_constant(self.X_loc)[0].shape[1] if self.constant else self.X_loc.shape[1]
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bws_same_times = bws_same_times
        self.verbose = verbose

        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min * k
        else:
            raise AttributeError(
                "multi_bw_min must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max * k
        else:
            raise AttributeError(
                "multi_bw_max must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        if pool:
            warnings.warn("The pool parameter is no longer used and will have no effect; parallelization is default and implemented using joblib instead.", RuntimeWarning, stacklevel=2)

        
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_multi = init_multi
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params['search_method'] = search_method
        self.search_params['criterion'] = criterion
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        self.search_params.update(init_multi=init_multi, tol_multi=tol_multi,
            rss_score=rss_score, max_iter_multi=max_iter_multi,
            multi_bw_min=multi_bw_min, multi_bw_max=multi_bw_max,
            bws_same_times=bws_same_times)
        
        self.int_score = not self.fixed

        if self.multi:
            self._mbw()
            self.params = self.bw[3]  
            self.sel_hist = self.bw[5] 
            self.bw_init = self.bw[6]  
        else:
            self._bw()
            self.sel_hist = self.bw[-1]
            
        
        return self.bw[0]

    def _bw(self):
        sgwr_func = lambda bw: getDiag[self.criterion](SGWR(
            self.coords, self.y, self.X_loc, self.data, bw, self.bt_value, self.att_bw, family=self.family, kernel=
            self.kernel, fixed=self.fixed, constant=self.constant, offset=self.
            offset, spherical=self.spherical, n_jobs=self.n_jobs).fit(lite=True))

        self._optimized_function = sgwr_func

        if self.search_method == 'golden_section':
            a, c = self._init_section(self.X_glob, self.X_loc, self.coords,
                                      self.constant)
            delta = 0.38197  
            self.bw = golden_section(a, c, delta, sgwr_func, self.tol,
                                     self.max_iter, self.bw_max, self.int_score,
                                     self.verbose)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                                     sgwr_func, self.int_score, self.verbose)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(
                self.X_glob, self.X_loc, self.coords, self.constant)
            if self.bw_min == self.bw_max:
                raise Exception(
                    'Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.'
                )
            objective = (lambda value: sgwr_func(np.round(value))) if self.int_score else sgwr_func
            self._optimize_result = minimize_scalar(
                objective, bounds=(self.bw_min, self.bw_max), method='bounded')
            optimum = np.round(self._optimize_result.x) if self.int_score else self._optimize_result.x
            score = sgwr_func(optimum)
            self.bw = [optimum, score, [(optimum, score)]]
        else:
            raise TypeError('Unsupported computational search method ',
                            self.search_method)

    def _mbw(self):
        y = self.y
        
        if self.constant:
            X,keep_x,warn = USER.check_constant(self.X_loc)
            data, keep_data,warn = USER.check_constant(self.data)
        else:
            X = self.X_loc
            data = self.data
        n, k = X.shape
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
        spherical = self.spherical
        coords = self.coords
        search_method = self.search_method
        criterion = self.criterion
        bw_min = self.bw_min
        bw_max = self.bw_max
        multi_bw_min = self.multi_bw_min
        multi_bw_max = self.multi_bw_max
        interval = self.interval
        tol = self.tol
        max_iter = self.max_iter
        bws_same_times = self.bws_same_times
        bt_value = self.bt_value
        #data = self.data
        alphacurve = self.alphacurve
        att_bw = self.att_bw
        if self.criterion == "CV":
            cv_criterion = True
        else:
            cv_criterion = False
    
        def sgwr_func(y, X, data, bw, bt_value, att_bw):
            return SGWR(coords, y, X, data, bw, bt_value, att_bw,
                       family=family, kernel=kernel, fixed=fixed, offset=offset,
                       constant=False, spherical=spherical, hat_matrix=False, n_jobs=self.n_jobs).fit(lite=True)

        def bw_func(y, X, data, bt_value=None, att_bw=None):
            selector = Sel_BW(coords=coords, y=y, X_loc=X, data=data,
                              X_glob=[], family=family, kernel=kernel, fixed=fixed,
                              offset=offset, constant=False, spherical=self.spherical, n_jobs=self.n_jobs)
            return selector
        
        def sel_func(bw_func, bw_min=None, bw_max=None): 
            return bw_func.search(
                    search_method=search_method, criterion=criterion, 
                    bw_min=self.bw_min if bw_min is None else bw_min,
                    bw_max=self.bw_max if bw_max is None else bw_max, interval=interval, tol=tol,
                    max_iter=max_iter, verbose=False)

        # self.bw = multi_bw(self.init_multi, y, X, n, k, coords, alphacurve, cv_criterion, fixed,
        #                    family, self.tol_multi, self.max_iter_multi, self.rss_score, sgwr_func,
        #                    bw_func, sel_func, multi_bw_min, multi_bw_max,
        #                    bws_same_times, verbose=self.verbose)

        self.bw = multi_bw(
                init=self.init_multi,
                y=y,
                X=X,
                data=data, ##3/27/26
                n=n,
                k=k,
                coords=coords,
                alphacurve=alphacurve,
                cv_criterion=cv_criterion,
                fixed=fixed,
                family=family,
                tol=self.tol_multi,
                max_iter=self.max_iter_multi,
                rss_score=self.rss_score,
                sgwr_func=sgwr_func,
                bw_func=bw_func,
                sel_func=sel_func,
                multi_bw_min=multi_bw_min,
                multi_bw_max=multi_bw_max,
                bws_same_times=bws_same_times,
                verbose=self.verbose,
                parameter=self.parameter,
                pmutation=self.pmutation,
                constant=self.constant
            )
        
    def local_cdist(self, coords_i, coords, spherical):
        """
        Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
        """
        if spherical:
            dLat = np.radians(coords[:, 1] - coords_i[1])
            dLon = np.radians(coords[:, 0] - coords_i[0])
            lat1 = np.radians(coords[:, 1])
            lat2 = np.radians(coords_i[1])
            a = np.sin(
                dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371.0
            return R * c
        else:
            return np.sqrt(np.sum((coords_i - coords)**2, axis=1))
        
    def _init_section(self, X_glob, X_loc, coords, constant):
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        if constant:
            n_vars = n_glob + n_loc + 1
        else:
            n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            min_dist = np.min(np.array([np.min(np.delete(
                self.local_cdist(coords[i],coords,spherical=self.spherical),i))
                    for i in range(n)]))
            max_dist = np.max(np.array([np.max(
                self.local_cdist(coords[i],coords,spherical=self.spherical))
                    for i in range(n)]))
                    
            a = min_dist / 2.0
            c = max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None and not np.isinf(self.bw_max):
            c = self.bw_max

        if a > c:
            raise ValueError('Minimum bandwidth exceeds maximum; provide bw_min or multi_bw_min for small samples.')
        return a, c

import numpy as np
from copy import deepcopy
from collections import defaultdict
import math 
import os
import pandas as pd

def golden_section(a, c, delta, function, tol, max_iter, bw_max, int_score=False,verbose=False):
    n = c if np.isfinite(c) else a
    if c == np.inf:
        b = a + delta * np.abs(n - a)
        d = n - delta * np.abs(n - a)
    else:
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
    
    if a == c:
        score = function(a)
        return a, score, [(a, score)]
    opt_val = a
    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    output = []
    dict = {}
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)

        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
            if verbose:
                print("Bandwidth: ", np.round(b, 2), ", score: ",
                      "{0:.2f}".format(np.asarray(score_b).item()))

        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d
            if verbose:
                print("Bandwidth: ", np.round(d, 2), ", score: ",
                      "{0:.2f}".format(np.asarray(score_d).item()))

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        output.append((opt_val, opt_score))
        
        opt_val = np.round(opt_val, 2)
        if (opt_val, opt_score) not in output:
            output.append((opt_val, opt_score))
        
        diff = score_b - score_d
        score = opt_score
        
    
    if a == np.inf or bw_max == np.inf:
        score_ols = function(np.inf)
        output.append((np.inf, score_ols))
            
        if score_ols <= opt_score:
            opt_score = score_ols
            opt_val = np.inf
        
        if verbose:
            print("Bandwidth: ", np.inf, ", score: ",
                    "{0:.2f}".format(np.asarray(score_ols).item()))

    return opt_val, opt_score, output

def alpha_fit(i, bw, bt_value, coords, y, X, data, att_bw, fixed=False, cv_criterion=False):
    from .weights import composite_weights
    wi = composite_weights(coords, i, bw, bt_value, data, fixed)
    xw = X.T * wi
    B = np.linalg.solve(xw @ X, xw)
    prediction = (X[i] @ B @ y).item()
    residual = y[i].item() - prediction
    influence = (X[i] @ B[:, i]).item()
    return residual ** 2, influence

def alpha_optimization(bw, bt_value, coords, y, X, data, n, att_bw, fixed=False, cv_criterion=False):
    if cv_criterion:
        RSS = 0.0
        trS = 0.0
        CVsum = 0.0
        eps = 1e-12
    
        for i in range(n):
            err2, s_ii = alpha_fit(i, bw, bt_value, coords, y, X, data, att_bw, fixed=fixed, cv_criterion=cv_criterion)
            RSS += err2
            trS += s_ii
            denom = max(1.0 - s_ii, eps)
            CVsum += err2 / (denom * denom)
    
        
        cv = CVsum / n
        return [cv]
    
    else:
        RSS = 0
        trS = 0
        for i in range(n):
            err2, hat = alpha_fit(i, bw, bt_value, coords, y, X, data, att_bw, fixed=fixed, cv_criterion=cv_criterion)
            RSS += err2
            trS += hat
    
        if n <= trS + 2:
            return [np.inf]
        aicc = n * np.log((RSS) / (n)) + n * np.log(2 * np.pi) + n * (n + trS) / (n - trS - 2.0)
        return [aicc]

def greedy_fit(coords, y, X, data, bw, n, k, fixed=False, alphacurve=False, cv_criterion=False):
    
    att_bw = 100  ## when we use either median as attribute bandwidth or the method for article groundwater (3rd law of geography)
        
    if alphacurve: 
        initial_candidates = [0.9, 0.01]
        alpha_scores = {}
        best_alpha = None
        best_score = float('inf')

        # Evaluate initial candidates
        for alpha in initial_candidates:
            if alpha < 0.02:
                continue
            aicc = alpha_optimization(bw, alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
            alpha_scores[alpha] = aicc
           
            if aicc < best_score:
                best_score = aicc
                best_alpha = alpha
            else:
                break

        def recursive_search(low, high, depth=0):
            if abs(high - low) < 0.01 or depth > 10:
                return

            mid = round((low + high) / 2, 3)
            if mid in alpha_scores or mid < 0.02:
                return

            aicc = alpha_optimization(bw, mid, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
            alpha_scores[mid] = aicc
            nonlocal best_alpha, best_score
            if aicc < best_score:
                best_alpha = mid
                best_score = aicc
                recursive_search(low, mid, depth + 1)
                recursive_search(mid, high, depth + 1)
            else:
                if mid < best_alpha:
                    recursive_search(mid, best_alpha, depth + 1)
                else:
                    recursive_search(best_alpha, mid, depth + 1)

        sorted_initial = sorted(initial_candidates)
        best_index = sorted_initial.index(best_alpha)
        if best_index > 0:
            recursive_search(sorted_initial[best_index - 1], best_alpha)
        if best_index < len(sorted_initial) - 1:
            recursive_search(best_alpha, sorted_initial[best_index + 1])
        
        if best_alpha==0.9:
            alpha =1
            aicc = alpha_optimization(bw, alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
            alpha_scores[alpha] = aicc
            if aicc < best_score:
                best_score = aicc
                best_alpha = alpha

            else:
                alpha =0.95
                aicc = alpha_optimization(bw, alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
                alpha_scores[alpha] = aicc
                if aicc < best_score:
                    best_score = aicc
                    best_alpha = alpha
                
        return best_alpha, att_bw, alpha_scores
        
    else:  
        initial_candidates = [0.7, 0.5, 0.1]
        alpha_scores = {}
        best_alpha = None
        best_score = float('inf')

        for alpha in initial_candidates:
            aicc = alpha_optimization(bw, alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
            alpha_scores[alpha] = aicc
            if np.isfinite(aicc) and aicc < best_score:
                best_score = aicc
                best_alpha = alpha
                
        if best_alpha is None:
            best_alpha = 0.5
            best_score = alpha_scores.get(best_alpha, float('inf'))

        def greedy_direction_search(start, direction, min_alpha=0.01):
            nonlocal best_alpha, best_score
            if start is None:
                return  

            current = start
            while True:
                step = 0.1 if current > 0.1 else 0.02
                next_alpha = round(current + direction * step, 3)

                if next_alpha < min_alpha or next_alpha > 1.0 or next_alpha in alpha_scores:
                    break

                aicc = alpha_optimization(bw, next_alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
                alpha_scores[next_alpha] = aicc

                if not np.isfinite(aicc):
                    break

                if aicc < best_score:
                    best_alpha = next_alpha
                    best_score = aicc
                    current = next_alpha
                else:
                    break

        greedy_direction_search(best_alpha, direction=-1)
        greedy_direction_search(best_alpha, direction=1)

        if best_alpha == 0.02:
            next_alpha = 0.0
            if next_alpha not in alpha_scores:
                aicc = alpha_optimization(bw, next_alpha, coords, y, X, data, n, att_bw, fixed=fixed, cv_criterion=cv_criterion)[0]
                alpha_scores[next_alpha] = aicc
                if np.isfinite(aicc) and aicc < best_score:
                    best_alpha = next_alpha
                    best_score = aicc
                    
        return best_alpha, att_bw

def multi_bw(init, y, X, data, n, k, coords, alphacurve, cv_criterion, fixed, family,
             tol, max_iter, rss_score, sgwr_func, bw_func, sel_func,
             multi_bw_min, multi_bw_max, bws_same_times, verbose=False,
             parameter=None, pmutation=False, constant=True):
    """
    Multiscale M-SGWR bandwidth search procedure using iterative GAM backfitting.

    If pmutation=True:
        - bandwidths are fixed from `parameter`
        - only alpha and att_bw are re-optimized for each bootstrap sample
    """
    
    opt_alpha = 1
    att_bw = 40 + 2 * k

    # Initial model
    if init is None:
        if pmutation:
            if parameter is None:
                raise ValueError("When pmutation=True, `parameter` must contain the original optimized bandwidths.")
            bw = np.asarray(parameter, dtype=float)
            bw_sgwr = float(bw[0])
            # use first bandwidth only for initialization fit
            optim_model = sgwr_func(y, X, data, bw[0], opt_alpha, att_bw)
        else:
            bw = sel_func(bw_func(y, X, data, opt_alpha, att_bw))
            bw_sgwr = bw
            optim_model = sgwr_func(y, X, data, bw, opt_alpha, att_bw)
    else:
        bw = init
        bw_sgwr = bw
        optim_model = sgwr_func(y, X, data, init, opt_alpha, att_bw)

    err = optim_model.resid_response.reshape((-1, 1))
    param = optim_model.params

    XB = np.multiply(param, X)
    if rss_score:
        rss = np.sum(err ** 2)

    scores = []
    delta = 1e6
    BWs = []
    bw_stable_counter = 0
    bws = np.empty(k)

    sgwr_sel_hist = []
    opt_alpha_vec = np.zeros(k)
    opt_att_bw = np.zeros(k)
    alpha_per_bw_hist = defaultdict(list)
    final_opt_bw_alpha = {}

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, desc=''):
            return x

    fixed_bws = None
    if pmutation:
        fixed_bws = np.asarray(parameter, dtype=float).copy()
        if fixed_bws.size != k:
            raise ValueError(f"`parameter` must have length {k}, but got {fixed_bws.size}.")

    att_bw = 40 + 2 * k
    
    for iters in tqdm(range(1, max_iter + 1), desc='Backfitting'):
        new_XB = np.zeros_like(X)
        params = np.zeros_like(X)

        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1)) + err
            temp_X = X[:, j].reshape((-1, 1))
            temp_data = data[:, j].reshape((-1, 1))

            # Step 1: bandwidth
            if pmutation:
                bw = fixed_bws[j]
            else:
                opt_alpha = 1
                bw_class = bw_func(temp_y, temp_X, temp_data, opt_alpha)

                if bw_stable_counter >= bws_same_times:
                    bw = bws[j]
                else:
                    bw = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j])
                    sgwr_sel_hist.append(deepcopy(bw_class.sel_hist))

            # Step 2: alpha
            if constant and j == 0:
                opt_alpha = 1
               
            else:
                if alphacurve:
                    opt_alpha, att_bw, alpha_scores = greedy_fit(
                        coords, temp_y, temp_X, temp_data, bw, n, k,
                        fixed=fixed, alphacurve=alphacurve,
                        cv_criterion=cv_criterion
                    )
                else:
                    opt_alpha, att_bw= greedy_fit(coords, temp_y, temp_X, temp_data, bw, n, k, fixed=fixed, alphacurve=alphacurve, cv_criterion=cv_criterion)
                    
            opt_alpha_vec[j] = opt_alpha
            opt_att_bw[j] = att_bw
            alpha_per_bw_hist[j].append((bw, opt_alpha, att_bw))

            # Step 3: fit local model using fixed or updated bw with optimized alpha
            optim_model = sgwr_func(temp_y, temp_X, temp_data, bw, opt_alpha, att_bw)
            err = optim_model.resid_response.reshape((-1, 1))
            param = optim_model.params.reshape((-1,))

            new_XB[:, j] = optim_model.predy.reshape(-1)
            params[:, j] = param
            bws[j] = bw

        if (iters > 1) and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0

        num = np.sum((new_XB - XB) ** 2) / n
        den = np.sum(np.sum(new_XB, axis=1) ** 2)
        score = (num / den) ** 0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy) ** 2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss

        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))

        if verbose:
            print("Current iteration:", iters, ", SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws]))
            print("Alphas:", ', '.join([str(np.round(a, 4)) for a in opt_alpha_vec]))

        if delta < tol:
            break

    opt_bws = fixed_bws if pmutation else BWs[-1]

    for j in range(k):
        final_opt_bw_alpha[j] = (opt_bws[j], opt_att_bw[j], opt_alpha_vec[j])

    return (
        opt_bws,
        np.array(BWs),
        np.array(scores),
        params,
        err,
        sgwr_sel_hist,
        bw_sgwr,
        alpha_per_bw_hist,
        final_opt_bw_alpha
    )

def equal_interval(a, c, interval, function, int_score=False, verbose=False):
    if a is None or c is None or interval <= 0 or a > c:
        raise ValueError('Interval search needs ordered bounds and a positive interval.')
    candidates = np.unique(np.r_[np.arange(a, c, interval), c])
    if int_score:
        candidates = np.unique(np.round(candidates))
    history = [(bw, function(bw)) for bw in candidates]
    bw, score = min(history, key=lambda item: item[1])
    return bw, score, history

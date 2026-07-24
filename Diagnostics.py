"""
Sourced from mgwr model which is authored by ((Taylor Oshan tayoshan@gmail.com)). However, minor modefication has been made to align with M-SGWR model. 
"""
__author__ = "M. Naser Lessani naserlessani252@gmail.com"

import numpy as np
from scipy import linalg
from spglm.family import Gaussian


def get_AICc(sgwr):
    """
    Get AICc value
    
    Gaussian: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    
    GWGLM: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)

    """
    n = sgwr.n
    k = sgwr.tr_S
    if isinstance(sgwr.family, Gaussian):
        aicc = -2.0 * sgwr.llf + 2.0 * n * (k + 1.0) / (
            n - k - 2.0)  
    return aicc


def get_AIC(sgwr):
    """
    Get AIC calue

    Gaussian: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)

    GWGLM:  AIC(G)=D(G) + 2K(G), where D and K denote the deviance and the effective
    number of parameters in the model with bandwidth G, respectively.
    
    """
    k = sgwr.tr_S
    y = sgwr.y
    mu = sgwr.mu
    if isinstance(sgwr.family, Gaussian):
        aic = -2.0 * sgwr.llf + 2.0 * (k + 1)
    return aic


def get_BIC(sgwr):
   
    n = sgwr.n  
    k = sgwr.tr_S
    y = sgwr.y
    mu = sgwr.mu
    if isinstance(sgwr.family, Gaussian):
        bic = -2.0 * sgwr.llf + (k + 1) * np.log(n)
    return bic


def get_CV(sgwr):
    
    aa = sgwr.resid_response.reshape((-1, 1)) / (1.0 - sgwr.influ)
    cv = np.sum(aa**2) / sgwr.n
    return cv


def corr(cov):
    invsd = np.diag(1 / np.sqrt(np.diag(cov)))
    cors = np.dot(np.dot(invsd, cov), invsd)
    return cors
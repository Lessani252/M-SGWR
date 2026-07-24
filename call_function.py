__author__ = "M. Naser Lessani naserlessani252@gmail.com"

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as tdist
from main_class import MSGWR
from bw_optimization import Sel_BW

df = pd.read_csv(r"directory\data")

columns = df.columns[3:]

g_y = df[df.columns[2]].values.reshape(-1, 1)   
g_x = df[columns].values                          

u = df[df.columns[0]].values                    
v = df[df.columns[1]].values                   

## If you decide to standardize your data prior to model calibration
scaler = StandardScaler()
g_x = scaler.fit_transform(g_x)
g_y = scaler.fit_transform(g_y)

g_coords = list(zip(u, v))

data = df[columns]  

## This option is able to more accurately optimize the alpha value but it's computationally demanding
selector = Sel_BW(g_coords, g_y, g_x, data, columns, multi=True, alphacurve=True)

## This option is faster but less acurate in terms of alpha optimization
# selector = Sel_BW(g_coords, g_y, g_x, data, columns, multi=True)

msgwr_bw = selector.search()

print('Optimal bandwidths:', msgwr_bw)
print('Optimal alphas:', np.array([v[-1] for v in selector.bw[-1].values()])) 

msgwr_res = MSGWR(g_coords, g_y, g_x, selector, data, columns).fit()

print('R2: ', msgwr_res.R2)
print('Adjusted R2: ', msgwr_res.adj_R2)
print('AICc: ', msgwr_res.aicc)
print('BIC: ', msgwr_res.bic)
print('RSS:', np.sum(msgwr_res.resid_response**2))


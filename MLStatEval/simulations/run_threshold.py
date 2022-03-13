# Script to generate figures and simulations for threshold calibration on single performance measure

import os
import numpy as np
import pandas as pd
import plotnine as pn
from funs_stats import get_CI
from utils import gg_save
from funs_threshold import tools_thresh

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')

# Different bootstrapping methods
di_method = {'point':'Naive', 'basic':'BS-Basic', 'quantile':'BS-Quantile', 'bca':'BS-BCa'}
di_msr = {'sens':'Sensitivity', 'spec':'Specificity'}


#######################################
# ---- (1) THRESHOLD AS QUANTILE ---- #

# (i) Run the simulation
nsim = 20000
nchunk = 2500
niter = nsim // nchunk
n = 100
gamma = 0.8
fmt_gamma = '%i%%' % (gamma*100)
mu = 1
enc_thresh = tools_thresh('sens', 'spec', mu=mu)
holder = []
for i in range(niter):
    print('--- Iteration %i of %i ---' % (i+1, niter))
    enc_thresh.sample(n=n, k=nchunk, seed=i)
    enc_thresh.learn_thresh(gamma=gamma)
    enc_thresh.thresh2oracle()
    enc_thresh.thresh2emp()
    enc_thresh.merge_dicts()
    tmp_df = enc_thresh.df_res.merge(enc_thresh.thresh_gamma)
    holder.append(tmp_df)
df_res = pd.concat(holder).reset_index(drop=True).drop(columns='cidx')
del holder

# (ii) Coverage by method
df_res = df_res.assign(cover = lambda x: np.where(x['oracle'] >= gamma,'>','<'))
df_res = df_res.merge(enc_thresh.thresh_gamma)
df_res['method'] = df_res['method'].map(di_method)
df_res['m'] = df_res['m'].map(di_msr)

# (iii) Make labels
cn_gg = ['m', 'method', 'cover', 'thresh_gamma']
df_text = df_res.groupby(cn_gg).size().reset_index().set_index(cn_gg).sort_index(ascending=False)
df_text.rename(columns={0:'n'}, inplace=True)
df_text = get_CI(df_text, 'n', nsim).assign(n=lambda x: x['n']/nsim)
df_text = (df_text*100).round(1).astype(str).reset_index()
# Make fancy text
df_text['lbl'] = df_text.apply(lambda x: '%s%s\n%s%% (%s, %s)' % (x['cover'], fmt_gamma, x['n'], x['lb'], x['ub']), 1)
df_text.insert(0, 'y', nsim/3.75)
df_text = df_text.assign(sign1=lambda x: np.where(x['m']=='Specificity',-1,+1))
df_text = df_text.assign(sign2=lambda x: np.where(x['cover']=='<',+1,-1))
df_text = df_text.assign(x=lambda x: x['thresh_gamma'] + 0.95*x['sign1']*x['sign2'])
# For vlines
df_thresh = df_text.groupby(['m','method','thresh_gamma']).size().reset_index().drop(columns=[0])

# Plot
n_method = df_res['method'].unique().shape[0]
n_msr = df_res['m'].unique().shape[0]
width = n_msr*4.0
height = n_method*3.25
tmp_res = df_res.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
tmp_thresh = df_thresh.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
tmp_text = df_text.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
gg_quant = (pn.ggplot(tmp_res, pn.aes(x='thresh')) + pn.theme_bw() + 
    pn.labs(x='Empirically chosen threshold',y='Simulation frequency') + 
    pn.geom_histogram(color='grey',fill='grey',alpha=0.5,bins=30) + 
    pn.facet_grid('method~m',scales='free_x') + 
    pn.geom_vline(pn.aes(xintercept='thresh_gamma'), data=tmp_thresh, inherit_aes=False) + 
    pn.guides(color=False) + 
    pn.geom_text(pn.aes(x='x', y='y', label='lbl',color='cover'), size=9, data=tmp_text, inherit_aes=False) + 
    pn.scale_x_continuous(limits=[-1.5, +2.5]))
gg_save('gg_quant.png', dir_figures, gg_quant, width, height)


##################################
# ---- (X) DIABETES DATASET ---- #

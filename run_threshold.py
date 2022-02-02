# Script to generate figures and simulations for threshold calibration on single performance measure

import os
import numpy as np
import pandas as pd
import plotnine as pn
from funs_support import gg_save
from funs_stats import get_CI
from funs_threshold import tools_thresh

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')

# Different bootstrapping methods
di_method = {'point':'Naive', 'basic':'BS-Basic', 'quantile':'BS-Quantile', 'bca':'BS-BCa'}
di_msr = {'sens':'Sensitivity'}


#######################################
# ---- (1) THRESHOLD AS QUANTILE ---- #

# (i) Run the simulation
nsim = 2500
n = 100
seed = 1
gamma = 0.8
mu = 1
enc_thresh = tools_thresh('sens', mu=mu)
enc_thresh.sample(n=n, k=nsim, seed=seed)
enc_thresh.learn_thresh(gamma=gamma)
enc_thresh.thresh2oracle()
enc_thresh.thresh2emp()
enc_thresh.merge_dicts()

# (ii) Coverage by method
df_thresh_gamma = enc_thresh.thresh_gamma.assign(m=lambda x: x['m'].map(di_msr))
df_res = enc_thresh.df_res.copy()
df_res['method'] = pd.Categorical(df_res['method'],di_method).map(di_method)
df_res['m'] = df_res['m'].map(di_msr)
df_res = df_res.assign(tt = lambda x: np.where(x['oracle'] >= gamma, '>', '<'))
# (iii) Make labels
cn_gg = ['m', 'method', 'tt']
df_text = df_res.groupby(cn_gg).size().reset_index().set_index(cn_gg).sort_index(ascending=False)
df_text.rename(columns={0:'n'}, inplace=True)
df_text = get_CI(df_text, 'n', nsim).assign(n=lambda x: x['n']/nsim)
# Make fancy text
df_text = (df_text*100).round(1).astype(str).reset_index()
df_text = df_text.assign(lbl=lambda x: x['tt']+' '+str(int(gamma*100))+'%\n'+x['n']+'% ('+x['lb']+','+x['ub']+')')
# x-axis will be function of oracle 
df_text = df_text.merge(df_thresh_gamma).assign(y=nsim/5)
df_text = df_text.assign(x=lambda x: x['thresh_gamma'] + np.where(x['tt'] == '<', +0.8, -1.05))

# Plot
n_method = df_res['method'].unique().shape[0]
xlim = 1.5
gg_quant_sens = (pn.ggplot(df_res, pn.aes(x='thresh')) + pn.theme_bw() + 
    pn.labs(x='Empirically chosen threshold',y='Simulation frequency') + 
    pn.geom_histogram(color='grey',fill='grey',alpha=0.5,bins=30) + 
    pn.facet_grid('m~method') + 
    pn.geom_vline(pn.aes(xintercept='thresh_gamma'), data=df_thresh_gamma, inherit_aes=False) + 
    pn.scale_x_continuous(limits=[-xlim, +xlim]) + 
    pn.guides(color=False) + 
    pn.geom_text(pn.aes(x='x', y='y', label='lbl',color='tt'), data=df_text, inherit_aes=False))
gg_save('gg_quant_sens.png', dir_figures, gg_quant_sens, width=n_method*4.5, height=3.5)


#######################################
# ---- (2) SHOW CHOICE ON DIABETES DATASET ---- #

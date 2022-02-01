# Script to generate figures and simulations for threshold calibration on single performance measure

import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import norm
from funs_support import gg_save
from funs_stats import get_CI
from funs_threshold import tools_thresh

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')

#######################################
# ---- (1) THRESHOLD AS QUANTILE ---- #

nsim = 2500
n = 100
seed = 1
gamma = 0.8
mu = 1
enc_thresh = tools_thresh(m='sens', mu=mu)
enc_thresh.sample(n=n, k=nsim, seed=seed)
enc_thresh.learn_thresh(gamma, method='quantile')
enc_thresh.thresh2emp()
enc_thresh.thresh2oracle()
# Coverage by method....

df_quant = pd.DataFrame({'thresh':enc_thresh.thresh, 'emp':enc_thresh.perf_emp, 'oracle':enc_thresh.perf_oracle})
df_quant = df_quant.assign(tt=lambda x: np.where(x['oracle'] < gamma, '<','>'))

thresh_oracle = mu + norm.ppf(1-gamma)
print(np.mean(df_quant['thresh'] > thresh_oracle))
df_text = df_quant.groupby('tt').size().reset_index().set_index('tt').sort_index(ascending=False)
df_text.rename(columns={0:'n'}, inplace=True)
df_text = get_CI(df_text, 'n', nsim).assign(n=lambda x: x['n']/nsim)
df_text = (df_text*100).round(1).astype(str).reset_index()
df_text = df_text.assign(lbl=lambda x: 'Sensitivity '+x['tt']+' '+str(int(gamma*100))+'%\n'+x['n']+'% ('+x['lb']+','+x['ub']+')')
df_text = df_text.assign(x=[-0.25,0.60], y=nsim/11)

# Plot
gg_quant_sens = (pn.ggplot(df_quant, pn.aes(x='thresh')) + pn.theme_bw() + 
    pn.labs(x='Empirically chosen threshold',y='Simulation frequency') + 
    pn.geom_histogram(color='grey',fill='grey',alpha=0.5,bins=30) + 
    pn.geom_vline(xintercept=thresh_oracle) + 
    pn.guides(color=False) + 
    pn.geom_text(pn.aes(x='x', y='y', label='lbl',color='tt'), data=df_text))
gg_save('gg_quant_sens.png', dir_figures, gg_quant_sens, 4.5, 3.5)


#######################################
# ---- (2) SHOW CHOICE ON DIABETES DATASET ---- #

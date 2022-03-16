"""
Script to compare how different methods do in terms of estimating a conservative threshold and 
"""

# External modules
import os
import pathlib
import numpy as np
import pandas as pd
import plotnine as pn

# Internal modules
from trialML.trial import classification
from trialML.theory import gaussian_mixture
from trialML.utils.stats import get_CI
from trialML.utils.utils import gg_save
from trialML.utils.m_classification import lst_method

# Set up foldres
dir_here = pathlib.Path(__file__).parent
dir_figures = os.path.join(dir_here, 'figures')
# dir_figures = os.path.join(os.getcwd(),'trialML','simulations','figures')

# Labels for methods
di_method = {'point':'Point', 'basic':'Classical', 'percentile':'Percentile', 'bca':'BCa', 'umbrella':'NP Umbrella'}

###############################
# ----- (1) PARAMETERS ------ #

# Set up normal data-generating process
seed = 1
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
n_test = 100
k_exper = 5000
idx_exper = np.arange(k_exper)+1
normal_dgp = gaussian_mixture()
normal_dgp.set_params(p, mu1, mu0, sd1, sd0)
# Generate test and trial data
y_test, s_test = normal_dgp.gen_mixture(n_test, k_exper, seed=seed)
y_trial, s_trial = normal_dgp.gen_mixture(n_test, k_exper, seed=seed+1)

# Set trial target parameters
alpha = 0.05  # type-I error (for either power or threshold)
gamma = 0.60  # target performance
spread = 0.1  # Null hypothesis spread
n_trial = 100  # Number of (class specific) trial samples
n_bs = 250  # Number of bootstrap iterations

# Loop over the difference performance measures
lst_m = ['sensitivity', 'specificity', 'precision']

# Pre-calculate oracle values
normal_dgp.set_gamma(gamma=gamma)
# Check that it aligns
for m, thresh in normal_dgp.oracle_threshold.items():
    normal_dgp.set_threshold(threshold=thresh)
    err_target = np.abs(gamma - normal_dgp.oracle_m[m])
    assert err_target < 1e-5, 'set_threshold did not yield expected oracle m!'

# Oracle values will be used for plotting later
vlines = pd.DataFrame.from_dict({k:[v] for k,v in normal_dgp.oracle_threshold.items()})
vlines = vlines.melt(None,None,'m','oracle')


###################################
# ----- (2) RUN EXPERIMENT ------ #

holder_thresh, holder_pval = [], []
for m in lst_m:
    print('--- Running simulations for %s ---' % m)
    calibration = classification(alpha=alpha, gamma=gamma, m=m)
    # (i) Learn threshold on test set data
    calibration.learn_threshold(y=y_test, s=s_test, method=lst_method, n_bs=n_bs, seed=seed, inherit=True)

    # (ii) Add on oracle value to threshold
    thresh_test = calibration.threshold_hat.assign(idx=idx_exper)
    thresh_test = thresh_test.melt('idx',None,'method','threshold')
    thresh_test = thresh_test.assign(coverage=lambda x: normal_dgp.check_threshold_coverage(x['threshold'], m))
    thresh_test.insert(0, 'm', m)
    holder_thresh.append(thresh_test)

    # (iii) Estimate power
    calibration.calculate_power(spread=spread, n_trial=n_trial, threshold=calibration.threshold_hat)
    power_test = calibration.power_hat.assign(idx=idx_exper)
    power_test = power_test.melt('idx',None,'method','power')

    # (iv) Generate performance measure and test statistic on trial
    m_trial, pval_trial = calibration.statistic(y=y_trial, s=s_trial, threshold=calibration.threshold_hat, pval=True)
    # Merge p-value with power estimate
    pval_trial = pval_trial.assign(idx=idx_exper)
    pval_trial = pval_trial.melt('idx',None,'method','pval')
    pval_trial = pval_trial.assign(reject=lambda x: x['pval'] < alpha)
    pval_trial = pval_trial.merge(power_test)
    pval_trial.insert(0, 'm', m)
    holder_pval.append(pval_trial)
    

################################
# ----- (2) MERGE & AGG ------ #

# (i) Concatenate over m
df_thresh = pd.concat(holder_thresh).reset_index(drop=True)
df_pval = pd.concat(holder_pval).reset_index(drop=True)

# (ii) Aggregate by [m, method] and generate CIs
cn_gg = ['m', 'method']
df_thresh_agg = df_thresh.groupby(cn_gg)['coverage'].mean().reset_index()
df_thresh_agg = df_thresh_agg.assign(den=k_exper,num=lambda x: x['coverage']*k_exper)
df_thresh_agg = get_CI(df_thresh_agg, cn_num='num', cn_den='den', alpha=alpha)
df_thresh_agg.drop(columns=['num', 'den'], inplace=True)
# Repeat for p-values
df_pval_agg = df_pval.groupby(cn_gg)[['reject','power']].mean().reset_index()
df_pval_agg = df_pval_agg.melt(cn_gg,None,'tt')
df_pval_agg = df_pval_agg.assign(den=k_exper,num=lambda x: x['value']*k_exper)
df_pval_agg = get_CI(df_pval_agg, cn_num='num', cn_den='den', alpha=alpha)
df_pval_agg.drop(columns=['num', 'den'], inplace=True)


#####################################
# ----- (3) THRESOLD & POWER ------ #

fmt_gamma = '%i%%' % (gamma*100)

# (i) Get threshold position by coverage ,'coverage'
pos_txt = df_thresh.query('coverage==True').groupby(['m'])['threshold'].quantile([0.05,0.95])
pos_txt = pos_txt.reset_index().rename(columns={'level_1':'moment'})
pos_txt = pos_txt.merge(vlines).assign(err=lambda x: (x['threshold']-x['oracle']).abs())
pos_txt = pos_txt.loc[pos_txt.groupby('m')['err'].idxmax()]
pos_txt = pos_txt.reset_index(drop=True).drop(columns=['oracle','err','moment'])
pos_txt.rename(columns={'threshold':'x'}, inplace=True)
pos_txt.insert(0, 'y', k_exper / 3)
pos_txt = pos_txt.merge(df_thresh_agg, 'right', 'm')
pos_txt['lbl'] = pos_txt.apply(lambda x: '%0.1f%% (%0.1f-%0.1f%%)' % (100*x['coverage'], 100*x['lb'], 100*x['ub']), 1)

# (ii) Plot
n_method = df_thresh_agg['method'].unique().shape[0]
n_msr = df_thresh_agg['m'].unique().shape[0]
width = n_msr*3.5
height = n_method*2.5

# Order categories
tmp_thresh = df_thresh.assign(method=lambda x: pd.Categorical(x['method'], list(di_method)))
tmp_txt = pos_txt.assign(method=lambda x: pd.Categorical(x['method'], list(di_method)))
# (i) Threshold and oracle
gg_threshold_method = (pn.ggplot(tmp_thresh, pn.aes(x='threshold')) + pn.theme_bw() + 
    pn.labs(x='Empirically chosen threshold',y='Simulation frequency') + 
    pn.geom_histogram(color='grey',fill='grey',alpha=0.5,bins=30) + 
    pn.facet_grid('method~m',scales='free_x', labeller=pn.labeller(method=di_method)) + 
    pn.geom_vline(pn.aes(xintercept='oracle'), data=vlines, inherit_aes=False) + 
    pn.guides(color=False) + 
    pn.geom_text(pn.aes(x='x', y='y', label='lbl'), size=9, data=tmp_txt, inherit_aes=False) + 
    pn.scale_y_continuous(limits=[-1, +k_exper*0.4]) + 
    pn.scale_x_continuous(limits=[-1.5, +2.5]))
gg_save('gg_threshold_method.png', dir_figures, gg_threshold_method, width, height)

# (ii) Expected power and actual
tmp_pval = df_pval_agg.assign(method=lambda x: pd.Categorical(x['method'],list(di_method)).map(di_method))
tmp_pval['tt'] = tmp_pval['tt'].map({'reject':'Reject Null', 'power':'Expected power'})

width = n_msr * 3
height = 3
posd = pn.position_dodge(0.5)
gg_power_comp = (pn.ggplot(tmp_pval, pn.aes(x='tt',y='value',color='method')) + 
    pn.theme_bw() + 
    pn.scale_color_discrete(name='Method') + 
    pn.facet_wrap('~m') + 
    pn.labs(y='Empirical/expected frequency') + 
    pn.theme(axis_title_x=pn.element_blank()) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'), position=posd) + 
    pn.geom_point(position=posd,size=2))
gg_save('gg_power_comp.png', dir_figures, gg_power_comp, width, height)


print('~~~ p2_threshold_power.py ~~~')
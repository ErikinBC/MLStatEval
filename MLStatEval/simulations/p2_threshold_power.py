"""
Script to compare how different methods do in terms of estimating a conservative threshold and 
"""

# External modules
import numpy as np
import pandas as pd

# Internal modules
from MLStatEval.trial import classification
from MLStatEval.theory import gaussian_mixture
from MLStatEval.utils.stats import get_CI
from MLStatEval.utils.m_classification import lst_method


###############################
# ----- (1) PARAMETERS ------ #

# Set up normal data-generating process
seed = 1
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
n_test = 100 #250
k_exper = 50 # 1000
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
n_bs = 25 # 1000  # Number of bootstrap iterations

# Loop over the difference performance measures
lst_m = ['sensitivity', 'specificity']

# Pre-calculate oracle values
normal_dgp.set_gamma(gamma=gamma)
# Check that it aligns
for m, thresh in normal_dgp.oracle_threshold.items():
    normal_dgp.set_threshold(threshold=thresh)
    err_target = np.abs(gamma - normal_dgp.oracle_m[m])
    assert err_target < 1e-5, 'set_threshold did not yield expected oracle m!'


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


#################################
# ----- (3) PLOT RESULTS ------ #

fmt_gamma = '%i%%' % (gamma*100)
# # (ii) Coverage by method
# df_res = df_res.assign(cover = lambda x: np.where(x['oracle'] >= gamma,'>','<'))
# df_res = df_res.merge(enc_thresh.thresh_gamma)
# df_res['method'] = df_res['method'].map(di_method)
# df_res['m'] = df_res['m'].map(di_msr)

# # (iii) Make labels
# cn_gg = ['m', 'method', 'cover', 'thresh_gamma']
# df_text = df_res.groupby(cn_gg).size().reset_index().set_index(cn_gg).sort_index(ascending=False)
# df_text.rename(columns={0:'n'}, inplace=True)
# df_text = get_CI(df_text, 'n', nsim).assign(n=lambda x: x['n']/nsim)
# df_text = (df_text*100).round(1).astype(str).reset_index()
# # Make fancy text
# df_text['lbl'] = df_text.apply(lambda x: '%s%s\n%s%% (%s, %s)' % (x['cover'], fmt_gamma, x['n'], x['lb'], x['ub']), 1)
# df_text.insert(0, 'y', nsim/3.75)
# df_text = df_text.assign(sign1=lambda x: np.where(x['m']=='Specificity',-1,+1))
# df_text = df_text.assign(sign2=lambda x: np.where(x['cover']=='<',+1,-1))
# df_text = df_text.assign(x=lambda x: x['thresh_gamma'] + 0.95*x['sign1']*x['sign2'])
# # For vlines
# df_thresh = df_text.groupby(['m','method','thresh_gamma']).size().reset_index().drop(columns=[0])

# # Plot
# n_method = df_res['method'].unique().shape[0]
# n_msr = df_res['m'].unique().shape[0]
# width = n_msr*4.0
# height = n_method*3.25
# tmp_res = df_res.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
# tmp_thresh = df_thresh.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
# tmp_text = df_text.assign(method=lambda x: pd.Categorical(x['method'],di_method.values()))
# gg_quant = (pn.ggplot(tmp_res, pn.aes(x='thresh')) + pn.theme_bw() + 
#     pn.labs(x='Empirically chosen threshold',y='Simulation frequency') + 
#     pn.geom_histogram(color='grey',fill='grey',alpha=0.5,bins=30) + 
#     pn.facet_grid('method~m',scales='free_x') + 
#     pn.geom_vline(pn.aes(xintercept='thresh_gamma'), data=tmp_thresh, inherit_aes=False) + 
#     pn.guides(color=False) + 
#     pn.geom_text(pn.aes(x='x', y='y', label='lbl',color='cover'), size=9, data=tmp_text, inherit_aes=False) + 
#     pn.scale_x_continuous(limits=[-1.5, +2.5]))
# gg_save('gg_quant.png', dir_figures, gg_quant, width, height)




print('~~~ End of run.py ~~~')
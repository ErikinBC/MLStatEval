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

####################################
# ----- (1) RUN EXPERIMENTS ------ #

# (i) Set up normal data-generating process
seed = 1
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
n_test = 250
k_exper = 1000
normal_dgp = gaussian_mixture()
normal_dgp.set_params(p, mu1, mu0, sd1, sd0)
y_test, s_test = normal_dgp.gen_mixture(n_test, k_exper, seed=1234)

# (ii) Set trial target parameters
alpha = 0.05  # type-I error (for either power or threshold)
gamma = 0.50  # target performance
spread = 0.1  # Null hypothesis spread
n_trial = 100  # Number of (class specific) trial samples
n_bs = 1000  # Number of bootstrap iterations

# Loop over the difference perfomrance measures
lst_m = ['sensitivity']  # , 'specificity'

di_n_trial = {'sensitivity':n_trial*p, 'specificity':n_trial*(1-p)}

holder_raw, holder_agg = [], []
for m in lst_m:
    print('--- Running simulations for %s ---' % m)
    calibration = classification(alpha=alpha, gamma=gamma, m=m)
    # (iii) Learn threshold on test set data
    calibration.learn_threshold(y=y_test, s=s_test, method=lst_method, n_bs=n_bs, seed=seed)
    # (iv) Estimate power
    n_trial_m = di_n_trial[m]  # E(no. of labels)
    calibration.calculate_power(spread=spread, n_trial=n_trial_m)

    # (v) Genereate trial data and evaluate
    y_trial, s_trial = normal_dgp.gen_mixture(n_test, k_exper, seed=seed+1)

    # Performance measure and test-statistic
    m_hat, pval = calibration.statistic(y=y_trial, s=s_trial, threshold=calibration.threshold_hat, pval=True)
    # Calculate whether test statistic was met
    m_hat = m_hat.melt(None,None,'method','val').assign(tt='stat')
    m_hat = m_hat.assign(ineq=lambda x: x['val'] > gamma)
    # Calculate whether null is rejected
    pval = pval.melt(None,None,'method','val').assign(tt='pval')
    pval = pval.assign(ineq=lambda x: x['val'] < alpha)
    res_m = pd.concat(objs=[m_hat, pval], axis=0)
    res_m.insert(0, 'm', m)
    holder_raw.append(res_m)
    # Get mean and number
    res_m_agg = res_m.groupby(['m','method','tt'])['ineq'].sum().reset_index()
    res_m_agg = res_m_agg.assign(n=k_exper).assign(rate=lambda x: x['ineq']/x['n'])
    # Merge on the expected power/coverage
    dat_oracle = pd.DataFrame({'tt':['stat', 'pval'], 'oracle':[1, calibration.power_hat]})
    res_m_agg = res_m_agg.merge(dat_oracle)
    holder_agg.append(res_m_agg)
    del res_m, res_m_agg


#################################
# ----- (2) PLOT RESULTS ------ #

res_raw = pd.concat(holder_raw).reset_index(drop=True)
res_agg = pd.concat(holder_agg).reset_index(drop=True)
res_agg = get_CI(df=res_agg, cn_num='ineq', cn_den='n', alpha=alpha)



# fmt_gamma = '%i%%' % (gamma*100)
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
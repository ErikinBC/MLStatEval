import os
import pathlib
import numpy as np
import pandas as pd
import plotnine as pn
import patchworklib as pw
from matplotlib import rc
# Internal
from trialML.utils.utils import makeifnot
from trialML.theory import gaussian_mixture
from trialML.utils.stats import emp_roc_curve, auc_rank

# Set up directories
dir_here = pathlib.Path(__file__).parent
dir_figures = os.path.join(dir_here, 'figures')
makeifnot(dir_figures)

# Tidy up the different labels
di_msr = {'sens':'Sensitivity', 'spec':'Specificity', 'thresh':'Threshold'}


############################################################
# --- (1) OPERATING THRESHOLD IS A RANDOM VARIABLE.... --- #

nsim, n_test = 250, 100
mu, p = 1, 0.5
ptail=1e-3
enc_dgp = gaussian_mixture()
enc_dgp.set_params(p=p, mu1=mu, mu0=0, sd1=1, sd0=1)
holder_auc = np.zeros(nsim)
holder_roc = []
# Gaussian mixture data
y, s = enc_dgp.gen_mixture(n=n_test, k=nsim, seed=nsim)
# Column-wise AUROC
emp_auroc = auc_rank(y, s)
holder_roc = []
for k in range(nsim): # Empirical ROC curve
    tmp_roc = emp_roc_curve(y[:,k], s[:,k]).assign(sim=k)
    holder_roc.append(tmp_roc)
emp_roc = pd.concat(holder_roc).reset_index(drop=True)
# Creat the ground-truth ROC
s_seq = np.linspace(*norm.ppf(ptail) * np.array([mu, -mu]), 100)
gt_roc = pd.DataFrame({'sim':-1, 'thresh':s_seq, 'sens':norm.cdf(mu-s_seq), 'spec':norm.cdf(s_seq)})
# Merge
df_roc = pd.concat(objs=[emp_roc, gt_roc], axis=0).reset_index(drop=True)
df_roc = df_roc.assign(tt=lambda x: np.where(x['sim']==-1,'Ground Truth','Simulation'))

# (iii) Operating threshold on x-axis
sens_target = 0.5
df_emp_thresh = df_roc.melt(['sim','thresh','tt'], None, 'msr', 'val')
df_emp_thresh['msr'] = df_emp_thresh['msr'].map(di_msr)

# (a) What threshold is chosen to get to sens_target?
thresh_sens = df_emp_thresh.query('sim>=0 & msr=="Sensitivity"').drop(columns=['msr','tt'])
thresh_spec = df_emp_thresh.query('sim>=0 & msr=="Specificity"').drop(columns=['msr','tt'])
thresh_chosen = interp_df(thresh_sens, 'val', 'thresh', sens_target, 'sim')
thresh_spec = thresh_spec.merge(thresh_chosen)
thresh_spec = thresh_spec.groupby('sim').apply(lambda x: interp_df(x, 'thresh', 'val', x['thresh_interp'].values[0], 'sim'))
thresh_spec.reset_index(drop=True, inplace=True)
thresh_spec = thresh_spec.rename(columns={'val_interp':'spec'}).assign(msr=di_msr['spec'])
thresh_chosen = thresh_chosen.rename(columns={'thresh_interp':'thresh'}).assign(msr=di_msr['sens'])
thresh_spec = thresh_spec.merge(thresh_chosen.drop(columns='msr'))
# Long-run sensitivity
thresh_chosen = thresh_chosen.assign(sens=lambda x: 1-norm(loc=mu).cdf(x['thresh']))

# (a) Intersection of thresholds to 50% sensitivity
gtit = 'A: Empirical threshold choice for %i%% sensitivity' % (100*sens_target)
gg_roc_thresh = (pn.ggplot(df_emp_thresh,pn.aes(x='val',y='thresh')) + 
    pn.theme_bw() + pn.labs(y='Operating threshold',x='Performance target') + 
    pn.ggtitle(gtit) + pn.facet_wrap('~msr') + 
    pn.geom_step(pn.aes(size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.geom_point(pn.aes(x=sens_target,y='thresh'),size=0.5,alpha=0.5,color='red',data=thresh_chosen) + 
    pn.geom_point(pn.aes(x='spec',y='thresh'),size=0.5,alpha=0.5,color='blue',data=thresh_spec) + 
    # pn.theme(legend_position=(0.5,-0.05),legend_direction='horizontal') + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.2]) + 
    pn.scale_size_manual(name=' ',values=[1.5,0.5]))
# gg_save('gg_roc_thresh.png', dir_figures, gg_roc_thresh, 9, 3.5)

# (b) Distribution of thresholds
gg_dist_thresh = (pn.ggplot(thresh_chosen, pn.aes(x='thresh')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='red',bins=20) + 
    pn.labs(x='Threshold',y='Frequency',title='B: Threshold distribution'))
# gg_save('gg_dist_thresh.png', dir_figures, gg_dist_thresh, 4.5, 3.5)

# (c) Distribution of long-run sensitivity
rc('text', usetex=True)
gg_dist_sens = (pn.ggplot(thresh_chosen, pn.aes(x='sens')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='green',bins=20) + 
    pn.geom_vline(xintercept=sens_target) + 
    pn.ggtitle('C: Oracle sensitivity') + 
    pn.labs(x='$\Phi(\mu - \hat{t})$',y='Frequency'))
# gg_save('gg_dist_sens.png', dir_figures, gg_dist_sens, 4.5, 3.5)
rc('text', usetex=False)

g1 = pw.load_ggplot(gg_roc_thresh, figsize=(9,3))
g2 = pw.load_ggplot(gg_dist_thresh, figsize=(4.5,3.5))
g3 = pw.load_ggplot(gg_dist_sens, figsize=(4.5,3.5))

gg_roc_process = g1 / (g2 | g3)
grid_save('gg_roc_process.png', dir_figures, gg_roc_process)


###################################
# --- (3) SENSITIVITY TARGET  --- #

# Tidy tt labels
di_tt = {'oracle':'Oracle', 'emp':'Empirical', 'trial':'Trial'}
nsim = 250
mu, p = 1, 0.5
n_test, n_trial = 200, 200
sens = 0.5
enc_dgp = dgp_bin(mu=mu, p=p, sens=sens)

holder = []
for i in range(nsim):
    test_i = enc_dgp.dgp_bin(n=n_test, seed=i)
    enc_dgp.learn_threshold(y=test_i['y'], s=test_i['s'])
    enc_dgp.create_df()
    trial_i = enc_dgp.dgp_bin(n=n_trial, seed=i+1)
    res_trial = enc_dgp.get_tptn(trial_i['y'], trial_i['s'])
    res_trial = res_trial.assign(tt='trial').melt('tt',None,'msr','val')
    tmp_df = pd.concat(objs=[enc_dgp.df_rv, res_trial], axis=0)
    tmp_df.insert(0, 'sim', i)
    holder.append(tmp_df)
res_tfpr = pd.concat(holder).reset_index(drop=True)
res_tfpr['msr'] = res_tfpr['msr'].map(di_msr)
res_tfpr['tt'] = res_tfpr['tt'].map(di_tt)
res_tfpr_wide = res_tfpr.pivot_table('val',['sim','msr'],'tt').reset_index()
res_tfpr_wide = res_tfpr_wide.melt(['sim','msr',di_tt['oracle']])
res_tfpr_wide = res_tfpr_wide.dropna().reset_index(drop=True)

# (i) Oracle vs OOS vs Empirical
gg_sens_scatter = (pn.ggplot(res_tfpr_wide, pn.aes(x='value',y='Oracle',color='tt')) + 
    pn.theme_bw() + pn.scale_color_discrete(name=' ') + 
    pn.labs(x='Observed',y='Oracle') + 
    pn.geom_point() + pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.facet_wrap('~msr'))
# gg_sens_scatter.save(os.path.join(dir_figures,'gg_sens_scatter.png'),height=4,width=12)

# (ii) Histogram distribution
gg_sens_hist = (pn.ggplot(res_tfpr, pn.aes(x='val',fill='tt')) + 
    pn.theme_bw() + pn.scale_fill_discrete(name=' ') + 
    pn.labs(x='Value',y='Frequency') + 
    pn.geom_histogram(alpha=0.5,color='grey',bins=30,position='identity') + 
    pn.facet_wrap('~msr',scales='free_x'))
# gg_sens_hist.save(os.path.join(dir_figures,'gg_sens_hist.png'),height=4,width=12)


####################################
# --- (4) COMPARATIVE STATICS  --- #

method = 'quantile'
lst_margin = np.arange(0.01, 0.31, 0.01)
lst_n_trial = [50, 100, 250, 1000]
lst_n_test = [250, 500, 1000]
# Use a single instance
enc_dgp = dgp_bin(mu=1, p=0.5, sens=0.5, alpha=0.05)
holder_static = []
for n_test in lst_n_test:
    dat_test = enc_dgp.dgp_bin(n=n_test, seed=1)
    enc_dgp.learn_threshold(y=dat_test['y'], s=dat_test['s'])
    df_emp = pd.DataFrame({'msr':['sens','spec'], 'emp':[enc_dgp.sens_emp, enc_dgp.spec_emp]})
    null_sens = enc_dgp.sens_emp - lst_margin
    null_spec = enc_dgp.spec_emp - lst_margin
    enc_dgp.set_null_hypotheis(null_sens, null_spec)
    for n_trial in lst_n_trial:
        n1_trial, n0_trial = n_trial, n_trial        
        enc_dgp.run_power('both', n1_trial, n0_trial, method, seed=1, n_bs=1000)
        # Calculate the margin
        tmp_df = enc_dgp.df_power.merge(df_emp).assign(n_test=n_test)
        tmp_df = tmp_df.assign(margin=lambda x: x['emp'] - x['h0'])
        holder_static.append(tmp_df)
# Merge and plot
res_static = pd.concat(holder_static).reset_index(drop=True)
cn_ns = ['n_trial', 'n_test']
res_static[cn_ns] = res_static[cn_ns].apply(pd.Categorical)
res_static['msr'] = res_static['msr'].map(di_msr)

# Plot
height = 3 * len(res_static['msr'].unique())
width = 3 * len(res_static['n_trial'].unique())
gg_power_margin = (pn.ggplot(res_static, pn.aes(x='margin',fill='n_test')) + 
    pn.theme_bw() + 
    pn.scale_fill_discrete(name='# of test samples') + 
    pn.facet_grid('msr~n_trial',labeller=pn.labeller(n_trial=pn.label_both)) + 
    pn.ggtitle('95% CI based on quantile approach') +
    pn.labs(x='Null hypothesis margin',y='Power estimate (CI)') + 
    pn.geom_ribbon(pn.aes(ymin='lb',ymax='ub'),color='black',alpha=0.5))
gg_power_margin.save(os.path.join(dir_figures,'gg_power_margin.png'),height=height,width=width)


############################
# --- (5) ROC & POWER  --- #

n = 100
n1_trial, n0_trial = 50, 50
method, n_bs = 'quantile', 1000
seed = 1
enc_dgp = dgp_bin(mu=1, p=0.5, thresh=0)

# (i) Generate data
dat_i = enc_dgp.dgp_bin(n, seed=seed)
t_lo, t_high = dat_i['s'].describe()[['min','max']]
thresh_seq = np.linspace(t_lo, t_high, 100)

# (ii) Get power
holder_thresh, holder_power = [], []
for j, thresh in enumerate(thresh_seq):
    # (i) Manually assign threshold
    enc_dgp.thresh_oracle = thresh
    enc_dgp.learn_threshold(dat_i['y'], dat_i['s'])
    null_sens = np.linspace(0.01, enc_dgp.sens_emp, 100)[:-1]
    null_spec = np.linspace(0.01, enc_dgp.spec_emp, 100)[:-1]
    enc_dgp.set_null_hypotheis(null_sens, null_spec)
    # (ii) Empirically interpolated threshold
    lst_sens = [enc_dgp.sens_emp, enc_dgp.sens_oracle]
    lst_spec = [enc_dgp.spec_emp, enc_dgp.spec_oracle]
    lst_oracle = ['Empirical', 'Oracle']
    res_thresh = pd.DataFrame({'thresh':thresh, 'boundary':lst_oracle, 'sens':lst_sens, 'spec':lst_spec})
    holder_thresh.append(res_thresh)
    # (iii) Power range
    enc_dgp.run_power('both', n1_trial, n0_trial, method, seed=seed, n_bs=n_bs)
    res_j = enc_dgp.df_power.assign(thresh=thresh)
    res_j.drop(columns=['method','n_trial'], inplace=True)
    holder_power.append(res_j)
# Merge and plot
df_thresh = pd.concat(holder_thresh).melt(['thresh','boundary'],None,'msr')
df_power = pd.concat(holder_power).rename(columns={'power':'oracle'})
df_power = df_power.melt(['msr','h0','thresh'],['oracle','lb','ub'],'tt','power')

di_tt = {'oracle':'Oracle', 'lb':'Lower-bound CI', 'ub':'Upper-bound CI'}
# (i) Plot the trade-off curve
gg_thresh_power = (pn.ggplot(df_power,pn.aes(x='thresh',y='h0',color='power')) + 
    pn.theme_bw() + pn.geom_point(alpha=0.5) + 
    pn.labs(x='Operating threshold',y='Null hypothesis') + 
    pn.facet_grid('msr~tt',labeller=pn.labeller(tt=di_tt,msr=di_msr)) + 
    pn.ggtitle('Black line shows empirical operating threshold and performance') + 
    pn.geom_line(pn.aes(x='thresh',y='value', linetype='boundary'), data=df_thresh, inherit_aes=False,size=1.0) + 
    pn.scale_linetype_discrete(name='Boundary') + 
    pn.scale_color_gradient2(name='Power',low='blue',mid='grey',high='red',midpoint=0.25))
gg_save('gg_thresh_power.png', dir_figures, gg_thresh_power, 9, 6)

print('~~~ End of gen_figures.py ~~~')
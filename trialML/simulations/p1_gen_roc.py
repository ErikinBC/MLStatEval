# Script for generating plots related to ROC

import os
import pathlib
import numpy as np
import pandas as pd
import plotnine as pn
import patchworklib as pw
from scipy.stats import skewnorm, norm

# Internal modules
from trialML.theory import gaussian_mixture
from trialML.utils.utils import makeifnot, gg_save  # grid_save
from trialML.utils.stats import emp_roc_curve, auc_rank, find_auroc

# Folders relative to script
dir_here = pathlib.Path(__file__).parent
dir_figures = os.path.join(dir_here, 'figures')
makeifnot(dir_figures)
# dir_figures = os.path.join(os.getcwd(), 'trialML', 'simulations', 'figures')


############################
# --- (1) ROC EXAMPLES --- #

n1, n0 = 1000, 1000
labels = np.append(np.repeat(1,n1), np.repeat(0,n0))
auc_target = 0.75
skew_seq = [-4, 0, 4]
di_skew = dict(zip(skew_seq, ['Left-skew','No skew','Right skew']))
holder_dist, holder_roc = [], []
np.random.seed(n1)
for skew in skew_seq:
    skew_lbl = di_skew[skew]
    # Find AUROC equivalent mean
    mu_skew = find_auroc(auc=auc_target, skew=skew)
    dist_y1 = skewnorm(a=skew, loc=mu_skew, scale=1)
    dist_y0 = skewnorm(a=skew, loc=0, scale=1)
    scores = np.append(dist_y1.rvs(n1), dist_y0.rvs(n0))
    emp_auc = auc_rank(labels, scores)
    print('Skew = %s, AUROC=%0.3f, mu: %0.2f' % (skew_lbl, emp_auc, mu_skew))
    df_dist = pd.DataFrame({'skew':skew_lbl,'y':labels, 's':scores})
    df_roc = emp_roc_curve(labels, scores).assign(skew=skew_lbl)
    holder_dist.append(df_dist)
    holder_roc.append(df_roc)
# Merge
roc_skew = pd.concat(holder_roc).reset_index(drop=True)
dist_skew = pd.concat(holder_dist).reset_index(drop=True)
auroc_skew = dist_skew.groupby('skew').apply(lambda x: auc_rank(x['y'],x['s'])).reset_index()
auroc_skew.rename(columns={0:'auroc'}, inplace=True)

# (i) Empirical ROC curves and skew
gg_roc_skew = (pn.ggplot(roc_skew,pn.aes(x='1-spec',y='sens',color='skew')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.geom_text(pn.aes(x=0.25,y=0.95,label='100*auroc'),size=9,data=auroc_skew,format_string='AUROC={:.1f}%') + 
    pn.facet_wrap('~skew') + pn.geom_step() + 
    pn.theme(legend_position='none'))
gg_save('gg_roc_skew.png', dir_figures, gg_roc_skew, 9, 3)


# (ii) Empirical score distribution and skew
gg_dist_skew = (pn.ggplot(dist_skew,pn.aes(x='s',fill='y.astype(str)')) + 
    pn.theme_bw() + pn.labs(x='Scores',y='Frequency') + 
    pn.theme(legend_position=(0.3, -0.03), legend_direction='horizontal',legend_box_margin=0) + 
    pn.facet_wrap('~skew') + 
    pn.geom_histogram(position='identity',color='black',alpha=0.5,bins=25) + 
    pn.scale_fill_discrete(name='Label'))
gg_save('gg_dist_skew.png', dir_figures, gg_dist_skew, 9, 3)

# # Combine both plots
# h, w = 9, 3
# g1 = pw.load_ggplot(gg_roc_skew, figsize=(h,w))
# g2 = pw.load_ggplot(gg_dist_skew, figsize=(h,w))
# gg_roc_dist_skew = pw.vstack(g2, g1, margin=0.25, adjust=False)
# grid_save('gg_roc_dist_skew.png', dir_figures, gg_roc_dist_skew)


##############################################
# --- (2) CHECK EMPIRICAL TO GROUNDTRUTH --- #

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
df_auc = pd.DataFrame({'auc':emp_auroc,'gt':enc_dgp.auroc})

# (i) Empirical ROC to actual
gg_roc_gt = (pn.ggplot(df_roc,pn.aes(x='1-spec',y='sens',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.geom_step() + 
    pn.theme(legend_position=(0.7,0.3)) + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[2,0.5]))
gg_save('gg_roc_gt.png', dir_figures, gg_roc_gt, 5, 3.5)

# (ii) Empirical AUROC to actual
gg_auc_gt = (pn.ggplot(df_auc,pn.aes(x='auc')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='red',bins=30) + 
    pn.geom_vline(pn.aes(xintercept='gt'),color='black',size=2) + 
    pn.labs(x='AUROC',y='Frequency'))
gg_save('gg_auc_gt.png', dir_figures, gg_auc_gt, 5, 3.5)

# h, w = 5, 3.5
# g1 = pw.load_ggplot(gg_roc_gt, figsize=(h, w))
# g2 = pw.load_ggplot(gg_auc_gt, figsize=(h, w))
# gg_roc_emp = pw.hstack(g2, g1, margin=0.25, adjust=False)
# grid_save('gg_roc_emp.png', dir_figures, gg_roc_emp)

print('~~~ End of p1_gen_roc.py ~~~')
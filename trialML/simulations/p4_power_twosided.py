# Script to generate results for two-sided power calculations

# External modules
import os
import pathlib
import numpy as np
import pandas as pd
import plotnine as pn

# Internal modules
from trialML.utils.utils import cvec
from trialML.utils.stats import get_CI
from trialML.theory import gaussian_mixture
from trialML.power import twosided_classification
from trialML.utils.theory import power_binom

# Set up foldres
# dir_here = pathlib.Path(__file__).parent
# dir_figures = os.path.join(dir_here, 'figures')
dir_figures = os.path.join(os.getcwd(),'trialML','simulations','figures')


###############################
# ----- (1) PARAMETERS ------ #

# Gaussian dist
normal_dgp = gaussian_mixture()
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
normal_dgp.set_params(p, mu1, mu0, sd1, sd0)
# Hypothesis test
alpha = 0.05
k_exper = 2500
margin_seq = np.arange(0.01, 0.32, 0.02).round(2)
n_trial_seq = [50, 100, 250, 1000]
n_test_seq = [250, 500, 1000]
n_perm = len(margin_seq) * len(n_trial_seq) * len(n_test_seq)
gamma1 = 0.5  # Target sensitivity
gamma2 = None
m1 = 'sensitivity'
m2 = 'specificity'
di_m = {1:m1, 2:m2}
di_p = {1:p, 2:1-p}   # Needs to align with m1/m2


###############################
# ----- (2) SIMULATION ------ #

# For columb subsetting
cn_merge = ['cidx', 'm']
cn_power = ['power_%s' % s for s in ['lb', 'ub', 'point']]

j = 0
holder = []
for margin in margin_seq:
    for n_test in n_test_seq:
        for n_trial in n_trial_seq:
            j += 1
            print('Iteration %i of %i' % (j, n_perm))
            # i) Generate test data
            y_test, s_test = normal_dgp.gen_mixture(n=n_test, k=k_exper, seed=j)
            # iia) Set up power calculator
            power_calc = twosided_classification(m1=m1, m2=m2, alpha=alpha)
            # iib) Learn thresold
            power_calc.set_threshold(y=y_test, s=s_test, gamma1=gamma1, gamma2=gamma2)
            # iic) Estimate power
            res_power = power_calc.get_power(n_trial=n_trial, margin=margin)
            # iii) Get oracle power
            normal_dgp.set_threshold(power_calc.threshold)
            oracle_m = pd.concat(objs=[normal_dgp.oracle_m[m1], normal_dgp.oracle_m[m2]])
            oracle_m = cvec(oracle_m.values)
            gamma0 = cvec(res_power['gamma0'].values)
            # Adjust for the fact that p will vary
            n_trial_eff = cvec(res_power['m'].map(di_p) * n_trial)
            oracle_power = power_binom(spread=margin, n_trial=n_trial_eff, gamma=oracle_m, gamma0=gamma0, alpha=alpha).flatten()
            # Store
            res_power = res_power.assign(oracle_power=oracle_power)
            res_power = res_power.assign(margin=margin, n_test=n_test, n_trial=n_trial)
            holder.append(res_power)
# Merge            
res_sim = pd.concat(holder).reset_index(drop=True)


####################################
# ----- (3) ANALYSIS & PLOT ------ #

cn_idx = ['n_test', 'n_trial', 'margin']
cn_gg = ['m'] + cn_idx

# (i) Average out power range of simulations #
res_sim_mu = res_sim.groupby(cn_gg)[cn_power].mean().reset_index()
res_sim_mu[cn_idx[:2]] = res_sim_mu[cn_idx[:2]].apply(pd.Categorical)
res_sim_mu['m'] = res_sim_mu['m'].astype(str)
di_m2 = {str(k):v for k,v in di_m.items()}

height = 3 * len(res_sim_mu['m'].unique())
width = 3 * len(res_sim_mu['n_trial'].unique())
gg_power_margin = (pn.ggplot(res_sim_mu, pn.aes(x='margin',fill='n_test')) + 
    pn.theme_bw() + 
    pn.scale_fill_discrete(name='# of test samples') + 
    pn.facet_grid('m~n_trial',labeller=pn.labeller(n_trial=pn.label_both, m=di_m2)) + 
    pn.labs(x='Null hypothesis margin',y='Power CI') + 
    pn.geom_ribbon(pn.aes(ymin='power_lb',ymax='power_ub'),color='black',alpha=0.5))
gg_power_margin.save(os.path.join(dir_figures,'gg_power_margin.png'),height=height,width=width)


# (ii) Calculate power coverage
res_cover = res_sim.assign(cover_lb=lambda x: (x['power_lb'] <= x['oracle_power']), cover_ub=lambda x: (x['power_ub'] >= x['oracle_power']))
res_cover = res_cover.assign(cover=lambda x: x['cover_lb'] & x['cover_ub'])
res_cover_mu = res_cover.groupby(cn_gg)['cover'].sum().reset_index()
res_cover_mu = res_cover_mu.assign(rate=lambda x: x['cover']/k_exper, den=k_exper)
res_cover_mu = get_CI(res_cover_mu, 'cover', 'den')
res_cover_mu[cn_gg] = res_cover_mu[cn_gg].apply(lambda x: pd.Categorical(x.astype(str), np.sort(x.unique()).astype(str)))

res_cover_mu['m'] = res_cover_mu['m'].map(di_m2)

height = 2.25 * len(res_sim_mu['n_test'].unique())
width = 2.75 * len(res_sim_mu['n_trial'].unique())
posd = pn.position_dodge(0.25)
gg_power_coverage = (pn.ggplot(res_cover_mu, pn.aes(x='margin',y='rate',color='m')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.labs(x='Null hypothesis margin', y='Coverage') + 
    pn.geom_hline(yintercept=1-alpha, linetype='--') + 
    pn.theme(axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(limits=[0.9,1.0],breaks=[0.9, 0.95, 1.0]) + 
    pn.facet_grid('n_test~n_trial',labeller=pn.label_both) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd))
gg_power_coverage.save(os.path.join(dir_figures,'gg_power_coverage.png'),height=height,width=width)

print('~~~ End of p4_power_twosided.py ~~~')
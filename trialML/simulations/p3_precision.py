# Script to generate plots and explore various aspects of precision
import os
import pathlib
import numpy as np
import pandas as pd
import plotnine as pn

# Internal modules
from trialML.theory import gaussian_mixture
from trialML.utils.utils import gg_save, df_float2int
from trialML.utils.theory import precision_threshold_range

# https://arxiv.org/pdf/1810.08635.pdf
# https://icml.cc/Conferences/2009/papers/309.pdf
# https://arxiv.org/pdf/1206.4667.pdf

# Set up directories
dir_here = pathlib.Path(__file__).parent
dir_figures = os.path.join(dir_here, 'figures')

##################################
# ---- (1) PRECISION CURVES ---- #

# Provide some examples of precision/threshold curves based on different parameter combinations

mu1, mu0 = 1, 0
p_seq = [0.1, 0.25, 0.5]
sd0_seq = [1, 2, 6]
sd1_seq = [1, 2, 6]
cn_gg = ['p','sd0', 'sd1']
dist = gaussian_mixture()
holder_pr, holder_points = [], []
for p in p_seq:
    for sd0 in sd0_seq:
        for sd1 in sd1_seq:
            dist.set_params(p, mu1, mu0, sd1, sd0)
            points = precision_threshold_range(mu1, mu0, sd1, sd0, p)
            points = np.atleast_2d((p, sd0, sd1) + points)
            tmp_pr_curve = dist.gen_pr_curve().assign(p=p, sd0=sd0, sd1=sd1)
            holder_points.append(points)
            holder_pr.append(tmp_pr_curve)
res_ppv = pd.concat(holder_pr).reset_index(drop=True)
res_ppv[cn_gg] = res_ppv[cn_gg].astype(str)
# Combine points
res_points = pd.DataFrame(np.vstack(holder_points).astype(float))
res_points = res_points.dropna().reset_index(drop=True)
res_points.columns = cn_gg + ['prec_min', 'prec_max', 'thresh']
df_float2int(res_points)
res_points = res_points.assign(y=lambda x: np.where(x['prec_min'] > 0, x['prec_min'], x['prec_max']))
res_points.drop(columns=['prec_min','prec_max'], inplace=True)
res_points[cn_gg] = res_points[cn_gg].astype(str)


# Create fancy labels
di_sd1 = dict(zip(pd.Series(sd1_seq).astype(str), ['$\sigma_1=%i$' % s1 for s1 in sd1_seq]))
di_sd0 = dict(zip(pd.Series(sd0_seq).astype(str), ['$\sigma_0=%i$' % s0 for s0 in sd0_seq]))

width = len(sd0_seq) * 3.5
height = len(sd1_seq) * 2.5
gg_ppv = (pn.ggplot(res_ppv,pn.aes(x='thresh',y='ppv',color='p')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.labs(x='Operating threshold',y='Precision') + 
    pn.scale_color_discrete(name='P(y=1)') + 
    pn.theme(panel_spacing_x=0.25, panel_spacing_y=0.1) + 
    pn.geom_point(pn.aes(x='thresh',y='y'),data=res_points) + 
    pn.facet_grid('sd1~sd0',labeller=pn.labeller(sd1=di_sd1, sd0=di_sd0),scales='free_x'))
gg_save('gg_ppv.png', dir_figures, gg_ppv, width, height)


print('~~~ End of p3_precision.py ~~~')
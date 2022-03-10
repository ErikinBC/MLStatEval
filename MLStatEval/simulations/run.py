# Run main classes (for quick debugging)

from MLStatEval.trial import classification
from MLStatEval.theory import gaussian_mixture
from MLStatEval.utils.performance import lst_method

# (i) Set up normal data-generating process
seed = 1
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
n, k = 250, 10
normal_dgp = gaussian_mixture()
normal_dgp.set_params(p, mu1, mu0, sd1, sd0)
y_sim, s_sim = normal_dgp.gen_mixture(n, k, seed=1234)
y, s = normal_dgp.gen_mixture(n, k, seed=seed)

# (ii) Set trial target parameters
alpha = 0.05  # type-I error (for either power or threshold)
gamma = 0.50  # target performance
spread = 0.1  # Null hypothesis spread
n_bs = 1000  # Number of bootstrap iterations

# Loop over the difference perfomrance measures
lst_m = ['sensitivity', 'specificity']

for m in lst_m:
    calibration = classification(alpha=alpha, gamma=gamma, m=m)
    # (iii) Learn threshold on data
    calibration.learn_threshold(y=y_sim, s=s_sim, method=lst_method, n_bs=n_bs, seed=seed)
    # (iv) Estimate power
    # calibration.estimate_power()

    # (v) Genereate test set of data and evaluate
    y_test, s_test = normal_dgp.gen_mixture(n, k, seed=seed+1)

    # Statistic
    m_hat = calibration.statistic(y_test, s_test, calibration.threshold_hat)
    m_hat = m_hat.assign(sim=range(k),gamma=gamma,m=m)
    m_hat = m_hat.melt(['m','gamma','sim'],None,'method','val')
    m_hat = m_hat.assign(ineq=lambda x: x['val'] >= x['gamma'])

    # Test and power


# (vi) Performance and plot




print('~~~ End of run.py ~~~')
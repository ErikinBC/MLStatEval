# Run main classes (for quick debugging)

from MLStatEval.trial import classification
from MLStatEval.theory import gaussian_mixture

m = 'sensitivity'  # performance measure

# (i) Set up normal data-generating process
p = 0.5
mu1, mu0 = 1, 0
sd1, sd0 = 1, 1
n, k = 250, 10
normal_dgp = gaussian_mixture(m=m)
normal_dgp.set_params(p, mu1, mu0, sd1, sd0)
y_sim, s_sim = normal_dgp.gen_mixture(n, k, seed=1234)

# (ii) Set trial target parameters
alpha = 0.05  # type-I error
gamma = 0.80  # target performance
calibration = classification(alpha=alpha, gamma=gamma, m=m)

# (iii) Learn threshold on data
calibration.set_threshold(y=y_sim, s=s_sim, method='empirical')

# (iv) Estimate power
# calibration.estimate_power()


print('~~~ End of run.py ~~~')
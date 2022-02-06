import pandas as pd
from timeit import timeit
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize_scalar
from funs_stats import err_prec, err_prec2

##############################
# --- (1) FASTEST FINDER --- #

target, mu, p = 0.8, 1, 0.5
num = 500

# (i) Root finder
lst_method = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748', 'secant']
# Newton/Halley are slow

holder = []
for method in lst_method:
    print(method)
    stmt = 'root_scalar(f=err_prec, args=(target, mu, p, False), x0=0, x1=0.1, method="%s", bracket=(-10,10)).root' % method
    thresh = eval(stmt)
    time = timeit(stmt, number=num, globals=globals())
    tmp = pd.DataFrame({'method':method, 'thresh':thresh, 'time':time}, index=[0])
    holder.append(tmp)
res_root = pd.concat(holder).sort_values('time').reset_index(drop=True).assign(tt='root')

# (ii) Minimize scalar
lst_method = ['brent', 'bounded', 'golden']
holder = []
for method in lst_method:
    print(method)
    stmt = 'minimize_scalar(fun=err_prec, args=(target, mu, p, True), method="%s", bounds=(-10,10)).x' % method
    thresh = eval(stmt)
    time = timeit(stmt, number=num, globals=globals())
    tmp = pd.DataFrame({'method':method, 'thresh':thresh, 'time':time}, index=[0])
    holder.append(tmp)
res_scalar = pd.concat(holder).sort_values('time').reset_index(drop=True).assign(tt='scalar')

# (iii) Pick "best"
res_both = pd.concat(objs=[res_root, res_scalar]).sort_values('time').reset_index(drop=True)
method = res_both['method'][0]
print(res_both)
print('The fast root-finder is: %s' % method)





# def err_prec2(target, thresh, mu, p, square):
#     num = norm.cdf(mu - thresh)*p
#     den = num + norm.cdf(-thresh) * (1-p)
#     num2 = -p*norm.pdf(mu - thresh)
#     den2 = num2 - norm.pdf(-thresh) * (1-p)
#     grad = (num2*den + num*den2) / den**2
#     return -grad

# Script to generate figures and simulations for threshold calibration on single performance measure

import numpy as np
import pandas as pd
from funs_class_thresh import tools_thresh

self = tools_thresh(m='sens')
y, s = self.sample(n=100,k=250)
s[y == 1].mean()
s[y == 0].mean()

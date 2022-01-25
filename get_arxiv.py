import os
import arxiv
import numpy as np
import pandas as pd
from time import time

# Category list
lst_cat = ['stat.ML', 'cs.AI', 'cs.LG']
n_max = 900000
didx = '1900-01-01'

holder = []
for cat in lst_cat:
    print('--- Category: %s ---' % cat)
    query = 'cat:%s' % cat
    # Set up the search: np.inf
    search = arxiv.Search(query=query, max_results=n_max, sort_by=arxiv.SortCriterion.SubmittedDate)
    # Search over that category
    storage = pd.Series(pd.to_datetime(np.repeat(didx,n_max+1)))
    stime = time()
    for i, result in enumerate(search.results()):
        storage[i] = result.published 
        if (i + 1) % 100 == 0:
            dtime = time() - stime
            rate = (i+1) / dtime
            print('Iteration %i (%0.1f queries per second)' % (i+1, rate))
    # Remove unused categories
    storage = storage[storage != pd.to_datetime(didx)]
    res = pd.DataFrame({'date':storage, 'tt':cat})
    holder.append(res)
# Merge all and save
df_publish = pd.concat(holder).reset_index(drop=True)
df_publish.to_csv('df_publish.csv', index=False)

print('~~~ End of get_arxiv.py ~~~')
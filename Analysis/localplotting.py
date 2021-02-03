# Installs !cythonize -a -i *.pyx --force # this should be done for each user as different headers might be used for
# different py versions !pip install git+http://github.com/aarjaneiro/parallelqueue@cythonized !pip install bokeh
# !pip install bigjson

# %%

import matplotlib.pyplot as plt
from parallelqueue.monitors import *

import datahelpers

# %%

# ToOrder = [25, 50, 100, 500, 1000]
local = datahelpers.FullImport("Tsim", 0)

# %%

test = [datahelpers.EcdfOverTime(local, 1000, i) for i in range(30)]

# %%

for i in range(2):
    plt.plot(pd.DataFrame({t: v(1) for t, v in test[i].items()}, index=list(test[i].keys())))

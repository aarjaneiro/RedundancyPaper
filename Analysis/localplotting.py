# Installs !cythonize -a -i *.pyx --force # this should be done for each user as different headers might be used for
# different py versions !pip install git+http://github.com/aarjaneiro/parallelqueue@cythonized !pip install bokeh
# !pip install bigjson

##
import sys
from parallelqueue.monitors import *
import datahelpers
# ToOrder = [25, 50, 100, 500, 1000]
local = datahelpers.FullImport("Tsim", 0)


test = [datahelpers.EcdfOverTime(local, 1000, i) for i in range(30)]
# Extract test times (uniques)
times = []
for m in range(30):
    for t, _ in test[m].items():
        times.append(t)
times = list(set(times))
times.sort()
times = pd.Series(times) # drop repeats
delta = np.mean(abs(times.diff())) * 30 # differenced series avg. deviation

print(datahelpers.TimeAverage(test, times, delta))


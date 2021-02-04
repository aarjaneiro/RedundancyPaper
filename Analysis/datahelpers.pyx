#cython: language_level=3

import bigjson
import numpy as np
import pandas as pd
from collections import deque
import sys

sys.path.append(".")

cpdef dict retrieve(str name, int N, int rep=-1):
    """
    For name.json, for some N size, retrieves either a specific rep or all reps (as opposed to storing all in RAM).
    
    return
    -------
    Dict of form: {rep: {N: dataframe of results}}
    """
    cdef dict local
    cdef object j

    with open(f"{name}.json", "rb") as fs:
        j = bigjson.load(fs)
        if (rep < 0):
            return j[str(N)].to_python()
        else:
            return j[str(N)][str(rep)].to_python()

## Methods for cleaning data

cpdef dict cleanIndeces(dict fromJson, int rep=-1):
    """
    e.g., 
    
    local = cleanIndeces(retrieve("Tsim",0))    
    """
    cpdef dict localJson = {}
    if rep < 0:  # case 1: parse for arbitrary number of reps
        queue = deque(fromJson.items())
        del fromJson  # free memory
        while queue:  # shrinks as iterates
            k, v = queue.popleft()
            localJson[_EvalCarefully(k)] = parser(v)
        return localJson

    else:
        return parser(fromJson)

cdef _EvalCarefully(target):
    """
    type-inferred evaluation which assigns NaNs 0
    """
    try:
        ret = eval(target)
    except Exception as e:
        ret = 0
        print(e)

    return ret  # can be either int or float (auto)

cdef parser(dict localJson):
    cpdef dict ret = {}
    cpdef dict v, newVals
    cdef str k

    queue = deque(localJson.items())
    del localJson  # free memory
    while queue:  # shrinks as iterates
        k, v = queue.popleft()
        newkey = _EvalCarefully(k)
        within = deque(v.items())
        while within:
            newVals = {}
            m, r = within.popleft()
            newVals[_EvalCarefully(m)] = r
        ret[newkey] = newVals
    return ret

# Preferable over manual iteration out of scope
cpdef dict FullImport(str name, int N, int rep=-1):
    """
    Returns data with original numerical values (for storage, these values are cast to strings).
    """
    return cleanIndeces(retrieve(name, N, rep))

## Plotting tools -- Already quite fast

cpdef dict getPerN(dict result, int rep):
    return result[rep]

def EcdfOverTime(dict result, int N, int rep):
    """
    Returns ECDF values in a list for some rep and N. N here is the literal value in ToOrder vs index of the value as prior.
    """
    cdef dict vals = {}
    cdef object queue

    queue = deque(getPerN(result, rep).items())
    del result
    while queue:
        key, value = queue.popleft()
        try:
            zeroes = (N - len(value)) / N

            def Ecdf(m, zeroes=zeroes, Tslice=value):  # appends an ecdf function
                return zeroes + sum([1 for i in Tslice.values() if i <= m]) / N

            vals[key] = Ecdf
        except Exception as e:
            print(e)
            pass
    return vals
## TimeAverage methods

cdef _TAalgorithm(float t, test, times, float delta):
    cdef float c
    cdef int i
    consider = deque(times.where(abs(times - t) <= delta))
    localData = []
    while consider:
        c = consider.popleft()
        for i in range(30):
            try:
                localData.append(test[i][c])
            except:
                pass
    return localData

cpdef list TimeAverage(test, times, float delta):
    """
    Time-averaging across simulations:

    for $t \in [0,T] = [0,\tau_{1}] \cup [\tau_{1},\tau_{2}] \dots \cup [\tau_{S},T]$ being the sim time partitioned
    by events, for $\tau^{a}_{i}$ being some entry/exit time of one particular sim run $a$, search the other sims
    for (e.g., sim $b$) closest $\tau^{b}_{n}$ such that $\|\tau_{i}^{a} - \tau_{n}^{b}\| \leq \Delta$ for some chosen
    $\Delta$. Given parameters are equal across sims, find the $i$th tau first. If $n=i$ then we consider this point when
    time-averaging, otherwise see ordering w.r.t. $\tau_{i}^{a}$ to choose another to test. If none exist, pass sim in time
    average for this event-time.
    """
    cdef float t
    cdef object m
    return [np.mean([m(1) for m in _TAalgorithm(t, test, times, delta)]) for t in times]

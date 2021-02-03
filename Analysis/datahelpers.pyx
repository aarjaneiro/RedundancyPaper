#cython: language_level=3

import bigjson
from collections import deque

from bigjson.obj import Object

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
    cpdef dict v
    cdef str k

    queue = deque(localJson.items())
    del localJson  # free memory
    while queue:  # shrinks as iterates
        k, v = queue.popleft()
        newkey = _EvalCarefully(k)
        newVals = {_EvalCarefully(m): r for m, r in v.items()}
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
    for key, value in getPerN(result, rep).items():
        try:
            zeroes = (N - len(value)) / N

            def Ecdf(m, zeroes=zeroes, Tslice=value):  # appends an ecdf function
                return zeroes + sum([1 for i in Tslice.values() if i <= m]) / N

            vals[key] = Ecdf
        except Exception as e:
            print(e)
            pass
    return vals
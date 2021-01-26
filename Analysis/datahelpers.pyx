#cython: language_level=3

import bigjson

#NaN = nan # avoids a name error. Might need to be explicit?

cpdef dict retrieve(str name, int N, int rep=-1):
    """
    For name.json, for some N size, retrieves either a specific rep or all reps (as opposed to storing all in RAM).
    
    return
    -------
    Dict of form: {rep: {N: dataframe of results}}
    """
    cpdef dict local
    with open(f"{name}.json","rb") as fs:
        j=bigjson.load(fs)
        if (rep < 0 ):
            local = j[str(N)].to_python()
        else:
            local = j[str(N)][str(rep)].to_python()
    return local

## Methods for cleaning data

cpdef dict cleanIndeces(dict fromJson, int rep=-1):
    """
    e.g., 
    
    local = cleanIndeces(retrieve("Tsim",0))    
    """
    cpdef dict localJson = {}
    
    if (rep < 0): # case 1: parse for arbitrary number of reps
        for key in list(fromJson.keys()):
            localJson[_EvalCarefully(key)] = parser(fromJson[key])
    else:
        localJson = parser(fromJson)
        
    return localJson

        
cdef _EvalCarefully(target):
    """
    type-inferred evaluation which assigns NaNs 0
    """
    try: 
        ret = eval(target)
    except Exception as e:
        ret = 0
        print(e)
        
    return ret # can be either int or float (auto)
        
    
cdef parser(dict localJson):
    cpdef dict ret = {}
    cpdef dict v
    cdef str k
    for k, v in localJson.items():
        newkey = _EvalCarefully(k)
        newVals = {_EvalCarefully(m): r for m,r in v.items()}
        ret[newkey] = newVals
    return ret
        

# Preferable over manual iteration out of scope
cpdef dict FullImport(str name, int N, int rep=-1):
    """
    Returns data with original numerical values (for storage, these values are cast to strings).
    """
    return cleanIndeces(retrieve(name, N, rep))

## Plotting tools

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

            def Ecdf(m, zeroes=zeroes, Tslice=value): # appends an ecdf function
                return zeroes + sum([1 for i in Tslice.values() if i <= m]) / N

            vals[key] = Ecdf
        except Exception as e:
            print(e)
            pass
    return vals


    


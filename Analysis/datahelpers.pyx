#cython: language_level=3

import bigjson as bj

cpdef dict retrieve(str name, int N, int rep=-1):
    """
    For name.json, for some N size, retrieves either a specific rep or all reps (as opposed to storing all in RAM).
    
    return
    -------
    Dict of form: {rep: {N: dataframe of results}}
    """
    cpdef dict local
    with open(f"{name}.json","rb") as fs:
        j=bj.load(fs)
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
            localJson[eval(key)] = parser(fromJson[key])
    else:
        localJson = parser(fromJson)
        
    return localJson
            
        
cpdef dict parser(dict localJson):
    cpdef dict ret = {}
    cpdef dict v
    cpdef str k
    for k, v in localJson.items():
        newKey = eval(k)
        newVals = {eval(m): r for m,r in v.items()}
        ret[newKey] = newVals
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

def PrepPlot(dict result, int N, int rep):
    """
    Returns ECDF values in a list for some rep and N. N here is the literal value in ToOrder vs index of the value as prior.
    """
    cpdef list vals = []
    for value in getPerN(result, rep).values():
        try:
            zeroes = (N - len(value)) / N

            def Ecdf(m, zeroes=zeroes, Tslice=value): # appends an ecdf function
                return zeroes + sum([1 for i in Tslice.values() if i <= m]) / N

            vals.append(Ecdf)
        except Exception as e:
            print(e)
            pass
    return vals
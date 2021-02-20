#cython: language_level=3

import json as js
cpdef void fileMaker(list results,list ToOrder,str name="Tsim",int reps=30):
    """
    Serializes time data into a dictionary --> JSON(str(Data))
    """
    cdef dict ret, subret, mret, m
    cdef int N, i
    ret = {}
    for N in ToOrder:
        subret = {}
        for i in range(reps):
            mret = {}
            try:
                # WARNING: k can be NaN => must be cast to string
                for k, v in results[i][N]["ReplicaClassCounts"].items(): #<- note the i-N switch!
                    mret[str(k)] = eval(v.to_json()) # so as not to disturb structure 
            except Exception as e: 
                print(f"{e}@{N}-{i}")
                pass
            subret[i] = mret
        ret[N] = subret
    with open(f"{name}.json", mode="w") as fp:
        js.dump(ret,fp)
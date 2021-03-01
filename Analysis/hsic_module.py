#!/bin/python

import gc
import io
import multiprocessing as mp
import time
from contextlib import redirect_stdout
from parallelqueue.base_models import *
from parallelqueue.monitors import *

from redundancy import hsic

ncpus = mp.cpu_count()

# %%

# User-set params
simrep = int(input("Number of reps: "))  # set to cores utilized in parallel to limit amount held in memory
if simrep > ncpus: print("simrep should be <= to be safe")
ToOrder = [5, 50, 100, 500]  # N values to test
# WARNING: N values too high can r/esult in memory leaks!
seed = int(input("seed: "))
maxtime = float(input("maxtime: "))  # max time used across all sims
mintime = float(input("mintime: "))
rng = hsic.spanning_grid_uniform(maxtime - 1, mintime)  # times to sample
alpha = 0.1

# %%

print(f"Using ncpus={simrep}")


class Concurrent:
    def __init__(self, maxtime=1000, rho=0.9, d=2, r=2,
                 order=range(2, 1000, 2), seed=123):
        self.rhoJSim = None
        self.r = r
        self.order = order
        self.d = d
        self.rho = rho
        self.maxtime = maxtime
        self._ts = None
        self.seed = seed

    def WriteEach(self, simrep=1, of=["TSim"], ts=True):
        """                   v rep within run
        res['Thresh(2,2)'][0][0].keys()
                           ^run (N-size)
        Out[9]: dict_keys(['ReplicaSets', 'TimeQueueSize'])"""
        self._ts = ts
        self._sims = simrep
        labels = {self.RSim: f"Redundancy({self.d})", self.JSim: f"JSQ({self.d})",
                  self.TSim: f"Thresh({self.d},{self.r})"}
        self.res = {}
        for sim in [self.__getattribute__(i) for i in of]:
            print(f"Running {sim}")
            self.res[labels[sim]] = self.ParallelSim(sim)

    def DoEach(self, of=["TSim"], iters=1):
        self._ts = False
        labels = {self.RSim: f"Redundancy({self.d})", self.JSim: f"JSQ({self.d})",
                  self.TSim: f"Thresh({self.d},{self.r})"}
        self.res = {}
        for sim in [self.__getattribute__(i) for i in of]:
            print(f"Running {sim}")
            results = self.ParallelSim(sim)
            self.res[labels[sim]] = pd.DataFrame(results)


    def RSim(self, reps):
        mons = [TimeQueueSize]
        testvalues = []
        for N in (self.order):
            _sim = RedundancyQueueSystem(maxTime=self.maxtime, parallelism=N, seed=self.seed + 2331 * N + reps,
                                         d=self.d,
                                         Arrival=random.expovariate,
                                         AArgs=(self.rho * N) / self.d, Service=random.expovariate, SArgs=1,
                                         Monitors=mons)
            _sim.RunSim()
            testvalues.append(_sim.MonitorOutput)
        if not self._ts:
            return np.array(testvalues)
        else:
            return np.mean(testvalues)

    def JSim(self, reps):
        mons = [TimeQueueSize]
        testvalues = []
        for N in (self.order):
            _sim = JSQd(maxTime=self.maxtime, parallelism=N, seed=self.seed + 2331 * N + reps, d=self.d,
                        Arrival=random.expovariate,
                        AArgs=(self.rho * N) / self.d, Service=random.expovariate, SArgs=1,
                        Monitors=mons)
            _sim.RunSim()
            testvalues.append(_sim.MonitorOutput)
        if not self._ts:
            return np.array(testvalues)
        else:
            return np.mean(testvalues)

    def TSim(self, reps):
        mons = [TimeQueueSize]
        testvalues = []
        for N in (self.order):
            _sim = ParallelQueueSystem(maxTime=self.maxtime, parallelism=N, seed=self.seed + 2331 * N + reps,
                                       d=self.d,
                                       r=self.r,
                                       Arrival=random.expovariate,
                                       AArgs=(self.rho * N) / self.d, Service=random.expovariate, SArgs=1,
                                       Monitors=mons)
            _sim.RunSim()
            testvalues.append(_sim.MonitorOutput)
        if not self._ts:
            return np.array(testvalues)
        else:
            return np.mean(testvalues)

    def ParallelSim(self, sim):
        with mp.Pool(processes=ncpus) as p:
            res = p.map(sim, range(self._sims))
        return res

    def Results(self):
        return self.res


def SafeRun(maxtime=1000, rho=0.9, d=2, r=2, order=range(2, 20, 2),
            of="TSim", seed=123, simrep=1, ts=True):  # Throws out Concurrent when done
    """
    Parallelized simulations with lambda such that
    rho = (d*lambda)/(mu*N) and mu = 1
    <=> lambda = (N*rho)/d.
    """
    run = Concurrent(maxtime, rho, d, r, order, seed)
    run.WriteEach(of=[of], simrep=simrep, ts=ts)
    return run.Results()


# %%

def rearrange_for_test(results, which=0, ToOrder=ToOrder):
    per_queue = {q: [] for q in range(ToOrder[which])}
    for sim in range(simrep):
        sim_results = results[sim][0]["TimeQueueSize"]
        queue_times = {q: {} for q in range(ToOrder[which])}
        for time, value in sim_results.items():
            for queue in range(ToOrder[which]):
                queue_times[queue][time] = value[queue]

        for queue in range(ToOrder[which]):
            per_queue[queue].append(queue_times[queue])
    return per_queue


def arr_t(X):
    return np.array(X).transpose()


gc.collect()  # clear memory of clutter from imports


# %%

# Outputs in tex tabular style
def sig(a):
    if a < 0.01:
        return "***"
    elif a < 0.05:
        return "**"
    elif a < 0.1:
        return "*"
    return ""


with open(f"output/{maxtime}fintabular_{seed}_{rng[:3]}..._{alpha}:{time.time()}.txt", "w") as ls:
    ls.write(f"\hline \n")
    ls.write(f"$\\rho$ & $N$ & $r$ & $\\hat p(H_{'{a}'})$ \\\ \n")
    ls.write(f"\hline \n")
    ls.write(f"\hline \n")
    ls.flush()
    for r in [0.8, 0.9, 0.99]:  # values of rho to test
        for thresh in [1, 2]:
            for i in range(len(ToOrder)):
                results = SafeRun(of="TSim", order=[ToOrder[i]], maxtime=maxtime, rho=r, r=thresh, simrep=simrep,
                                  ts=False, seed=seed)
                results = results[list(results.keys())[0]]  #
                sim_per_queue = rearrange_for_test(results, i)
                del results
                gc.collect()
                f = io.StringIO()
                with redirect_stdout(f):
                    data_sim = [arr_t(hsic.time_sampler(sim_per_queue[m], rng)) for m in range(ToOrder[i])]
                    del sim_per_queue
                    test = hsic.dHSIC_resample_test(data_sim, 500)
                    assert test <= 1
                ls.write(
                    f"{r if i == 0 and thresh == 2 else ''} & {ToOrder[i]} & {thresh} & {round(test, 3) if test is not None else None} {sig(test)} \\\ \n")
                if i == 4 and thresh == 2:
                    ls.write(f"\hline \n")
                ls.flush()
                del data_sim
                gc.collect()
    ls.close()
exit()


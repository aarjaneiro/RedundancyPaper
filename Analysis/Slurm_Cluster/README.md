
Using SLURM
------------

Slurm is a workload manager used for allocating computing resources in HPC scenarios.
For info on using Slurm with UW Math's research clusters, see [this guide](https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/job-submit-commands-examples). One can modify the `.sh` script included in this directory for their own use as a starting point.


The submission script will install the needed dependencies, render the C script from the `.pyx` file, compile it, and then run the simulation.
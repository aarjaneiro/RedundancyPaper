#!/bin/bash
#SBATCH --job-name="parallelqueue"
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8000

module load anaconda3
virtualenv out
source out/env/bin/activate
pip3 install --no-index --upgrade pip
pip3 install joblib simpy pandas numpy ipython cython
echo installing parallelqueue@cythonized
pip3 install git+http://github.com/aarjaneiro/parallelqueue@cythonized
echo Now running the script...
cythonize -a -i *.pyx
python3 hpc_script.py
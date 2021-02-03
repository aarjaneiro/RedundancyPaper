#!/bin/bash
cythonize -i *.pyx --force
rm *.c
python hpc_script.py
rm *.so
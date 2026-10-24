@echo off

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set BLIS_NUM_THREADS=1

call .venv\Scripts\activate.bat

python main.py

pause


#--- .\.venv\Scripts\python.exe -m pip install -r requirements.txt

#---  .\.venv\Scripts\python.exe main.py
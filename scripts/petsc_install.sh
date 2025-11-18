# Installation script for PETSc

git clone https://gitlab.com/petsc/petsc.git
cd petsc

./configure --with-mpi-dir=/apps/mpich-3.2.1--gcc-9.1.0/ --with-fc=0 --with-batch

make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-debug all
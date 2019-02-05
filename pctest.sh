#!/usr/bin/env zsh

### Job name
#BSUB -J Poly1

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o /home/dh573265/montepython_polychord/chains/Poly1.%J.%I
#BSUB -u hooper@physik.rwth-aachen.de

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 30:00

### Request memory you need for your job in TOTAL in MB
#BSUB -M 12000

#BSUB -R "select[hpcwork]"

### Hybrid Job with <N> MPI Processes in groups to <M> processes per node
### #BSUB -n <N>
### #BSUB -R "span[ptile=<M>]"
#BSUB -n 4
#BSUB -R "span[ptile=1]"

### Request a certain node type
#BSUB -m mpi-s

### Use nodes exclusive
# #BSUB -x

### Each MPI process with T Threads
export OMP_NUM_THREADS=12

### Choose an MPI: either Open MPI or Intel MPI
### Use esub for Open MPI
#BSUB -a openmpi

module unload intel
module unload openmpi
module load gcc
module load openmpi
module load python

export WORKDIR=/home/dh573265/montepython_polychord

export LD_LIBRARY_PATH=/home/dh573265/software/PolyChord/PolyChord/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/MPI/openmpi-1.10.4/linux/gcc_4.8.5/lib/libmpi.so:$LD_PRELOAD
export LD_LIBRARY_PATH=/home/dh573265/software/MultiNest/lib:$LD_LIBRARY_PATH

#export PYTHONPATH=/home/dh573265/software/PolyChord/PolyChord

## source /home/dh573265/planck/plc-2.0/bin/clik_profile.zsh

source /lustren/hpcwork/dh573265/plc-2.0/bin/clik_profile.zsh

### Change to the work directory
cd $WORKDIR

### Execute your application
$MPIEXEC $FLAGS_MPI_BATCH python montepython/MontePython.py run -p base2015_ns.param -o chains/pc -m PC  &> chains/pc/output.log

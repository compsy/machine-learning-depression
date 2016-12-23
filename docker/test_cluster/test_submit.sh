#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=3
#SBATCH --ntasks=30
#SBATCH --job-name=python_mpi
#SBATCH --mem=1000

module load Python/3.5.1-foss-2016a
module load R/3.3.1-foss-2016a
module load OpenMPI/1.10.2-GCC-4.9.3-2.25
mpirun python ./python_mpi.py

gitdir="ICPE_machine_learning_workgroup"
cd $gitdir
# pip install --user -r requirements.txt
echo "Running testcluster.py"
export MPLBACKEND="agg"
export OMP_NUM_THREADS=23
mpirun python3 test_cluster.py
echo "Finished testcluster.py"


#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=3
#SBATCH --job-name=python_cpu
#SBATCH --mem=8000
module load Python/3.5.1-foss-2016a
module load R/3.3.1-foss-2016a
module load OpenMPI/1.10.2-GCC-4.9.3-2.25-CUDA-7.5.18
gitdir="~/ICPE_machine_learning_workgroup"
cd $gitdir/learner
mpirun python main.py

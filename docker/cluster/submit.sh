#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=23
####SBATCH --partition=short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peregrine@compsy.nl
#SBATCH --job-name=python_cpu
#SBATCH --mem=8000
module load Python/3.5.1-foss-2016a
module load R/3.3.1-foss-2016a
#module load matplotlib/1.5.1-foss-2016a-Python-3.5.1
#module load Tk/8.6.4-foss-2016a-libX11-1.6.3
#module load libX11/1.6.3-foss-2016a
gitdir="ICPE_machine_learning_workgroup"
cd $gitdir
# pip install --user -r requirements.txt
echo "Running main.py"
export MPLBACKEND="agg"
export OMP_NUM_THREADS=23
mpirun python3 main.py -c -p -n -f
echo "Finished main.py"

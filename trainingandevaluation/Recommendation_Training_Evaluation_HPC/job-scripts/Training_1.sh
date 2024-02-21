#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=thin
#SBATCH --nodes 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=38


#SBATCH --mail-type=BEGIN,END,TIME_LIMIT,TIME_LIMIT_90,FAIL
#SBATCH --mail-user=user@mail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

cp -r $HOME/Recommendation_Training_Evaluation_HPC/data/Training "$TMPDIR"


mkdir "$TMPDIR"/output_dir


for i in `seq 1 3`; do
  python $HOME/Recommendation_Training_Evaluation_HPC/scripts/Training.py "$TMPDIR"/Training "$TMPDIR"/output_dir "" "$i" &
done
wait


cp -r "$TMPDIR"/output_dir $HOME/Recommendation_Training_Evaluation_HPC/Models
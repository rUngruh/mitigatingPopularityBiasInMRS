#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --partition=thin
#SBATCH --nodes 1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=12


#SBATCH --mail-type=BEGIN,END,TIME_LIMIT,TIME_LIMIT_90,FAIL
#SBATCH --mail-user= user@mail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

cp -r $HOME/Recommendation_Training_Evaluation_HPC/data/Evaluation "$TMPDIR"
cp -r $HOME/Recommendation_Training_Evaluation_HPC/Models "$TMPDIR"

mkdir "$TMPDIR"/output_dir

for i in `seq 1 10`; do
  python $HOME/Recommendation_Training_Evaluation_HPC/scripts/Evaluator.py "$TMPDIR"/Evaluation "$TMPDIR"/output_dir "$TMPDIR"/Models 1 "$i" &
done
wait


cp -r "$TMPDIR"/output_dir $HOME/Recommendation_Training_Evaluation_HPC/Evaluation
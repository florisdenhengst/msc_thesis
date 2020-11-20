#!/bin/bash

CONDA_ENV="/home/fdhengst/venvs/conda/abstr-summ"

echo "Loading 2020 modules" &&
module load 2020 &&
echo "Loading Anaconda module" &&
module load Anaconda3/2020.02 &&
echo "Activating conda env" &&
conda activate "$CONDA_ENV" &&
echo "Done"

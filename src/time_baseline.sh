#!/bin/bash

DATASETS="PH2 RETINA BOWL2018 SARTORIUS KITTI BDD"

for DATASET in $DATASETS; do
    echo "DATASET=$DATASET"
    time python3 two-seg-baseline.py $DATASET ours
done


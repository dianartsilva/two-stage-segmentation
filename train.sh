#!/bin/bash
DATASETS="PH2 EVICAN RETINA KITTI"
for DATASET in $DATASETS; do
    python3 train.py $DATASET 0
    python3 train.py $DATASET 1
done

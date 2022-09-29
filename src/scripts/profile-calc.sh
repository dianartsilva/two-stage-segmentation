#!/bin/bash

DATASETS="PH2 RETINA BOWL2018 SARTORIUS KITTI BDD"
PATCHES="2 4 8 16"

for DATASET in $DATASETS; do
    for PATCH in $PATCHES; do
        echo "DATASET=$DATASET PATCH=$PATCH"
        python3 profile.py $DATASET ours --npatches $PATCH
        echo " "
    done
done


#!/bin/bash

DATASETS="PH2 RETINA BOWL2018 SARTORIUS KITTI BDD"
PATCHES="2"

for DATASET in $DATASETS; do
    for PATCH in $PATCHES; do
        echo "DATASET=$DATASET PATCH=$PATCH"
        time python3 two-seg.py $DATASET ours --npatches $PATCH
    done
done


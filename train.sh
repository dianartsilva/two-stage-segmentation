#!/bin/bash
NPATCHES="2 4 8 16"
for NPATCH in $NPATCHES; do
    echo "NPATCH=$NPATCH"
    python3 train.py PH2 1 --npatches $NPATCH
    python3 two-seg.py PH2 --npatches $NPATCH
done

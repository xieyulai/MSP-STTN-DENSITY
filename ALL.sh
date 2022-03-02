#!/bin/bash
RECORD=5712

DATASET=All

#python3 pre_main_short.py --mode train --record $RECORD --dataset_type $DATASET --patch_method STTN

python3 pre_main_short.py --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN


tail record/$RECORD/log.txt -n 1

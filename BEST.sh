#!/bin/bash
RECORD=5712
DATASET=All


python3 pre_main_short.py --best 1 --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN


tail record/$RECORD/log.txt -n 1

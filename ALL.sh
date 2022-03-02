#!/bin/bash
RECORD=1720

###TRAIN
python3 pre_main_short.py --mode train --record $RECORD 
###TEST
python3 pre_main_short.py --mode val --record $RECORD 


tail record/$RECORD/log.txt -n 1

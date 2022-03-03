#!/bin/bash
#DIM256
#RECORD=5720
#DI512M
RECORD=3711
python3 pre_main_short.py --best 1 --mode val --record $RECORD 
tail record/$RECORD/log.txt -n 1

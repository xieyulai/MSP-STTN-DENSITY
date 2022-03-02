#!/bin/bash
RECORD=5712
python3 pre_main_short.py --best 1 --mode val --record $RECORD 
tail record/$RECORD/log.txt -n 1

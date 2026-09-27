#!/bin/bash
cd "$(dirname "$0")/.."
nohup python3 -u cluster/scheduler.py >> runs/scheduler.log 2>&1 &

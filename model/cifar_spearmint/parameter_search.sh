#!/bin/bash

cd `dirname $0`

SPEARMINT=/home/ubuntu/lib/spearmint/spearmint/bin/spearmint

#${SPEARMINT} ./spearmint/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0

PYTHON_SPEARMINT=/home/ubuntu/lib/spearmint/spearmint/

PYTHONPATH=${PYTHON_SPEARMINT}:${PYTHONPATH} LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/lib:${LD_LIBRARY_PATH} /usr/bin/python ${PYTHON_SPEARMINT}/spearmint/main.py --driver=local --method=GPEIOptChooser --method-args=noiseless=1 ./spearmint/config.pb

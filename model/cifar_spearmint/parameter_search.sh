#!/bin/bash

cd `dirname $0`

SPEARMINT=/home/laket/lib/spearmint/spearmint/bin/spearmint

#${SPEARMINT} ./spearmint/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0

PYTHON_SPEARMINT=/home/laket/lib/spearmint/spearmint/

PYTHONPATH=${PYTHON_SPEARMINT}:${PYTHONPATH} LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} /usr/bin/python ${PYTHON_SPEARMINT}/spearmint/main.py --driver=local --method=GPEIOptChooser --method-args=noiseless=1 ./spearmint/config.pb

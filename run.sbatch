#!/bin/bash

#
# your xecutable here
#

if [ ! -d "./tys_raw" ]; then
    python download_agora.py
fi

python create_samples.py
python train_net.py
python test_net.py


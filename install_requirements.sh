#!/bin/bash

pip install -e .
pip install -r requirements.txt
# for sm86
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow-gpu==2.9.2

pip install sklearn

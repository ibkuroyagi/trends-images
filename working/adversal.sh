#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
python -u adversal_valid.py -f 30 -fold 0

# sbatch --gres=gpu:1 -c 4 ./adversal.sh
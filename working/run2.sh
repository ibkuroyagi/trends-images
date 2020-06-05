#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
python -u one_target.py -fno 0 -tid 2 -fold 0 -verbose

# sbatch --gres=gpu:1 -c 4 ./run2.sh
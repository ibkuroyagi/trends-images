#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
python -u predict_one.py -fno 0 -tid 0 -fold 0 -nw 4
python -u predict_one.py -fno 0 -tid 1 -fold 0 -nw 4
python -u predict_one.py -fno 0 -tid 2 -fold 0 -nw 4
python -u predict_one.py -fno 0 -tid 3 -fold 0 -nw 4
python -u predict_one.py -fno 0 -tid 4 -fold 0 -nw 4
python -u predict_one.py -fno 0 -tid 0 -fold 1 -nw 4
# sbatch --gres=gpu:1 -c 4 ./predict_one.sh
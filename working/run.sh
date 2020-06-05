#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
python -u one_target.py -fno 1 -tid 0 -nw 4
# python -u one_target.py -fno 2 -tid 2 -nw 4
# python -u one_target.py -fno 1 -tid 1 -nw 4
# python -u one_target.py -fno 2 -tid 2 -nw 4
# sbatch --gres=gpu:1 -c 4 ./run.sh
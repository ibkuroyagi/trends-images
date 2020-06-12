#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
for i in {1..4} ; do
    python -u one_target.py -fno 1 -tid ${i} -nw 4 -fold 0
done
# sbatch --gres=gpu:1 -c 4 ./run1.sh

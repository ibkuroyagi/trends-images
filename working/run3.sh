#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
for i in {0..4} ; do
    python -u ridge_feature_selection.py -tid ${i} -f 2 -np 10
done
# sbatch  -c 10 ./run3.sh
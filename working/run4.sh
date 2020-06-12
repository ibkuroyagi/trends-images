#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
for i in {0..4} ; do
    #python -u ridge_alpha.py -f 0 -tid ${i}
    python -u ridge_feature_selection.py -f 1 -tid ${i}
done
# sbatch -c 4 ./run4.sh
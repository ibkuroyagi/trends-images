#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
np=4
export OMP_NUM_THREADS=2
# 作成したPythonスクリプトを実行
# python -u ridge_feature_selection.py -tid 2 -f 2 -np ${np}
for i in {0..5} ; do
    python -u ridge_feature_selection.py -tid ${i} -f 2 -np ${np}
done
# sbatch -c 2 -n 4 ./run3.sh
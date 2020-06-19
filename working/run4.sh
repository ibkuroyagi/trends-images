#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行
export OMP_NUM_THREADS=2
np=4
f=3
for i in {0..4} ; do
    python -u ridge_feature_selection_2.py -f ${f} -tid ${i} -np ${np}
done
# sbatch -c 2 -n 4 ./run4.sh
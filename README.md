# TReNDS Neuroimaging

## explain these files
<pre>
~/working% tree -L 1
.
├── adversal-valid.ipynb
├── @eaDir
├── ensemble
├── ibuki-gpu-one-target-fnc-loading-3d-resnet-cnn.ipynb
├── logs
├── models
├── one_target_cv.py
├── one_target.py
├── output
├── pictures
├── predict_one.py
├── predict_one.sh
├── ridge_alpha.ipynb
├── ridge_alpha.py
├── ridge_feature_selection_2.py
├── ridge_feature_selection.ipynb
├── ridge_feature_selection.py
├── ridge_results
├── run1.sh
├── run2.sh
├── run3.sh
├── run4.sh
├── run_cv.sh
├── run.sh
├── submission
├── test_check.ipynb
└── trends-lofo-feature-importance.ipynb
</pre>

### ensemble
#### the results of ensemble data valid and test  
One of the most importnt file.   
you have to use these ridge predicted results ```/ensemble/pred_ridge_No{file_No}.csv```  
* how to load columns?
```
import json
json_path = f"ensemble/ridge_No{file_No}_columns.json"
with open(json_path) as f:
    df = json.load(f)
```

### logs
log of neural network 3DCNN and Adversal validation  
log_resnet10{target}_* .csv is results of 3DCNN to predict target  
adversal_* .csv is results of 3DCNN to predict adversal-validation


### models 
model of neural network 3DCNN and Adversal validation  
resnet10{target}_* .pth is results of 3DCNN to predict target  
adversal_* .pth is results of 3DCNN to predict adversal-validation  

### pictures
resluts of neural network as same as logs, but it is easier to understand the results.  
most of the results of these pitcures are made by neural network(3DCNN)  


### ridge_results
results of ridge regression
#### mean of No
No0. use all feature(fnc, loading)  
No1. fix std Forcibly  
No2. decrease feature backward process  (0:1404, 0:1403,..., 0:1)  
N03. decrease feature backward process  (0:1, 0:2,..., 0:1404)  

### submission 
submmison files

### run*.sh
you don't heve to care if it's a command for the lab.

### test_check.ipynb
One of the most importnt file.  
this file makes /ridge_results/pred_{target}_No{file_No}_CV.csv
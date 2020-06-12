
import numpy as np
import pandas as pd
import argparse
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-tid", "--target_id", type=int, default=0,
    help="target_id (default=0)")
parser.add_argument(
    "-f", "--feature_names", type=int, default=0)

args = parser.parse_args()
target_id = args.target_id
feature_names = args.feature_names
# feature_names = 0
# target_id = 2
targets = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
target = targets[target_id]
weights = [0.3, 0.175, 0.175, 0.175, 0.175]
weight = weights[target_id]
print(target, weight, "feature_names", {feature_names})
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()


# %%
# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1 / 600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# %%
# To check the best alpha
df_model = df.copy()
max_iter = 31
NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2020)

features = loading_features + fnc_features

overall_score = 0
y_oof = np.zeros((df.shape[0], max_iter))
y_preds = np.zeros((NUM_FOLDS, max_iter, df.shape[0]))
y_scores = np.zeros((max_iter, NUM_FOLDS))
scores = np.zeros(max_iter)  
alphas = []
for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
    alpha = 1e-6
    for i in range(max_iter):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
        train_df = train_df[train_df[target].notnull()]

        model = Ridge(alpha=alpha)
        model.fit(train_df[features].values, train_df[target].values)

        val_pred = model.predict(val_df[features])
        y_preds[f, i] = model.predict(test_df[features])
        
        y_oof[val_ind, i] = val_pred
        null_idx = val_df[target].notnull()
        y_scores[i, f] = metric(val_df[target][null_idx].values, val_pred[null_idx]) * weight
        if f == 0:
            alphas.append(alpha)
        alpha *= 2

for i in range(max_iter):
    null_idx = df[target].notnull().values
    y_oof_not_null = y_oof[null_idx, i]
    y_true = df[null_idx][target].values
    scores[i] = metric(y_oof_not_null, y_true) * weight


# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for f in range(NUM_FOLDS):
    plt.plot(alphas, y_scores[:, f], label=f"fold{f}")
plt.legend()
plt.xlabel('alpha')
plt.ylabel(f'score:{target}')
plt.xscale('log')
plt.grid(alpha=0.5)
plt.title(f'No:{feature_names}, {target} each fold')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(alphas, scores)
plt.xlabel('alpha')
plt.ylabel(f'score:{target}')
plt.xscale('log')
plt.grid(alpha=0.5)
best_idx = np.where(scores == scores.min())[0][0]
plt.title(f'best score CV:{scores[best_idx]:.5f}, alhpa:{alphas[best_idx]:.5f} in {target}')
plt.tight_layout()
plt.savefig(f"ridge_results/{target}_No{feature_names}_CV.png")

print(f'best score CV:{scores[best_idx]:.5f}, alhpa:{alphas[best_idx]:.5f} in {target}')
valid_df = pd.DataFrame(np.zeros((df.shape[0], 2)), columns=["Id", f"{target}_pred"])
valid_df['Id'] = df.Id
valid_df[f"{target}_pred"] = y_oof[:, best_idx]
valid_df.to_csv(f"ridge_results/val_{target}_No{feature_names}_CV.csv", index=False)
pred_df = pd.DataFrame(np.zeros((test_df.shape[0], 2)), columns=["Id", f"{target}_pred"])
pred_df['Id'] = test_df.Id
pred_df[f"{target}_pred"] = y_preds.mean(axis=0)[best_idx]
pred_df.to_csv(f"ridge_results/pred_{target}_No{feature_names}_CV.csv", index=False)
print('over')
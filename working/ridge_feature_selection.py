import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from time import time


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))


parser = argparse.ArgumentParser()
parser.add_argument(
    "-tid", "--target_id", type=int, default=0,
    help="target_id (default=0)")
parser.add_argument(
    "-f", "--feature_names", type=int, default=0)
parser.add_argument(
    "-np", "--n_process", type=int, default=5)

args = parser.parse_args()
target_id = args.target_id
feature_names = args.feature_names
n_process = args.n_process

targets = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
target = targets[target_id]
weights = [0.3, 0.175, 0.175, 0.175, 0.175]
weight = weights[target_id]
best_alpha = [0.00026, 0.0041, 0.0041, 0.00205, 0.00205]
alpha = best_alpha[target_id]
print(target, weight, "feature_names:", feature_names, "alpha:", alpha)
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()
FNC_SCALE = 1 / 600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


df_model = df.copy()
max_iter = 1403
NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2020)

features = loading_features + fnc_features
max_features = len(features)

ridge_beta = pd.DataFrame(np.zeros((max_iter, max_features + 2)), columns=features + ["intercept_", "alpha"])
alpha = 1e-6
alphas = []
for i in range(31):
    alphas.append(alpha)
    alpha *= 2

n_alpha = len(alphas)


def search_best_alpha(_features, alpha, j):
    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
        train_df = train_df[train_df[target].notnull()]

        model = Ridge(alpha=alpha)
        model.fit(train_df[_features].values, train_df[target].values)

        val_pred = model.predict(val_df[_features])
        _y_oof[val_ind, j] = val_pred
        _y_preds[f, j] = model.predict(test_df[_features])
        null_idx = val_df[target].notnull()
        _y_scores[j, f] = metric(val_pred[null_idx], val_df[target][null_idx].values) * weight
        if f == 0:
            _ridge_beta.loc[j, _features] = model.coef_
            _ridge_beta.loc[j, ["intercept_"]] = model.intercept_
            _ridge_beta.loc[j, ["alpha"]] = alpha
    null_idx = df[target].notnull()
    _scores[j] = metric(_y_oof[null_idx, j], df[target][null_idx].values) * weight
    return _y_oof[:, j], _y_preds[:, j], _y_scores[j], _scores[j], _ridge_beta.iloc[j].values


def wrap_search_best_alpha(args):
    return search_best_alpha(*args)


y_oof = np.zeros((df.shape[0], max_iter))
y_preds = np.zeros((df.shape[0], max_iter))
y_scores = np.zeros((max_iter, NUM_FOLDS))
scores = np.zeros(max_iter)
t_start = time()
for i in tqdm(range(max_iter)):
    _features = features[:max_features - i]
    _y_oof = np.zeros((df.shape[0], n_alpha))
    _y_preds = np.zeros((NUM_FOLDS, n_alpha, df.shape[0]))
    _y_scores = np.zeros((n_alpha, NUM_FOLDS))
    _scores = np.zeros(n_alpha)
    _ridge_beta = pd.DataFrame(np.zeros((n_alpha, max_features + 2)), columns=features + ["intercept_", "alpha"])
    job_args = [(_features, alphas[j], j) for j in range(31)]
    p = Pool(processes=n_process)
    for j, (__y_oof, __y_preds, __y_scores, __scores, __ridge_beta) in enumerate(p.imap(wrap_search_best_alpha, job_args)):
        _y_oof[:, j] = __y_oof
        _y_preds[:, j] = __y_preds
        _y_scores[j] = __y_scores
        _scores[j] = __scores
        _ridge_beta.iloc[j] = __ridge_beta
    p.close()
    p.join()
    best_alpha_idx = np.where(_scores == _scores.min())[0][0]
    y_oof[:, i] = _y_oof[:, best_alpha_idx]
    y_scores[i] = _y_scores[best_alpha_idx]
    scores[i] = _scores[best_alpha_idx]
    y_preds[:, i] = _y_preds[:, best_alpha_idx].mean(axis=0)
    ridge_beta.iloc[i] = _ridge_beta.iloc[best_alpha_idx]
    print(f"\n[INFO] i:{i}, score:{scores[i]:.5f}, alpha:{ridge_beta.loc[i, ['alpha']].values[0]:.5f}, total time:{time()-t_start:.1f}")

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
for f in range(NUM_FOLDS):
    plt.plot(np.arange(max_iter - 1, -1, -1), y_scores[:, f], label=f"fold{f}")
plt.legend()
plt.xlabel('n_feature')
plt.ylabel(f'score:{target}')
plt.grid(alpha=0.5)
plt.title(f'No:{feature_names}, {target} each fold')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(np.arange(max_iter), scores)
plt.xlabel('n_feature')
plt.ylabel(f'score:{target}')
plt.grid(alpha=0.5)
best_idx = np.where(scores == scores.min())[0][0]
plt.title(f'best score CV:{scores[best_idx]:.5f}, n_feature:{np.arange(max_iter)[best_idx]:d} in {target}')
plt.tight_layout()
plt.savefig(f"ridge_results/{target}_No{feature_names}_CV.png")

print(f'best score CV:{scores[best_idx]:.5f}, n_feature:{np.arange(max_iter)[best_idx]:d} in {target}')
valid_df = pd.DataFrame(np.zeros((df.shape[0], 2)), columns=["Id", f"{target}_pred"])
valid_df['Id'] = df.Id.values
valid_df[f"{target}_pred"] = y_oof[:, best_idx]
valid_df.to_csv(f"ridge_results/val_{target}_No{feature_names}_CV.csv", index=False)
pred_df = pd.DataFrame(np.zeros((test_df.shape[0], 2)), columns=["Id", f"{target}_pred"])
pred_df['Id'] = test_df.Id.values
pred_df[f"{target}_pred"] = y_preds[best_idx]
pred_df.to_csv(f"ridge_results/pred_{target}_No{feature_names}_CV.csv", index=False)

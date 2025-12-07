
# !pip install lightgbm fairlearn

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.3
SENSITIVE_ATTR = "sex"
np.random.seed(RANDOM_SEED)

X, y = load_diabetes(return_X_y=True, as_frame=True)
y = (y > y.median()).astype(int)

# Adding manual demographics
rng = np.random.default_rng(RANDOM_SEED)
n = len(X)
age = np.clip(rng.normal(55, 12, size=n), 18, 90)
sex = rng.choice(["female", "male"], size=n)
race = []
for yi in y:
    if yi == 1:
        race.append(rng.choice(["white", "black", "asian"], p=[0.50, 0.35, 0.15]))
    else:
        race.append(rng.choice(["white", "black", "asian"], p=[0.70, 0.15, 0.15]))

df = X.copy()
df["outcome"] = y
df["age"] = age
df["sex"] = sex
df["race"] = race

#Split
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["outcome"],
    random_state=RANDOM_SEED
)

feature_cols = train_df.select_dtypes(include=[np.number]).drop(columns=["outcome"]).columns

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
y_train = train_df["outcome"].values
y_test = test_df["outcome"].values
S_test = test_df[SENSITIVE_ATTR].values

# Baseline
X_train_base = X_train.copy()
X_test_base = X_test.copy()

# Demographic Inclusion
age_train = train_df[["age"]].values
age_test = test_df[["age"]].values
cat_cols = ["sex", "race"]
train_cat = pd.get_dummies(train_df[cat_cols], drop_first=True)
test_cat = pd.get_dummies(test_df[cat_cols], drop_first=True)
train_cat, test_cat = train_cat.align(test_cat, join="left", axis=1, fill_value=0)

X_train_aug = np.hstack([X_train, age_train, train_cat.values])
X_test_aug = np.hstack([X_test, age_test, test_cat.values])

# Age Ablation
X_train_no_age = np.hstack([X_train, train_cat.values])
X_test_no_age = np.hstack([X_test, test_cat.values])

# LightGBM Training
def train_lgb(Xtr, ytr, Xte, yte):
    train_set = lgb.Dataset(Xtr, label=ytr)
    test_set = lgb.Dataset(Xte, label=yte, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": RANDOM_SEED
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[test_set],
        num_boost_round=300
    )
    return model

lgb_base = train_lgb(X_train_base, y_train, X_test_base, y_test)
lgb_aug = train_lgb(X_train_aug, y_train, X_test_aug, y_test)
lgb_no_age = train_lgb(X_train_no_age, y_train, X_test_no_age, y_test)

# Predictions
y_prob_base = lgb_base.predict(X_test_base)
y_pred_base = (y_prob_base > 0.5).astype(int)
auc_base = roc_auc_score(y_test, y_prob_base)

y_prob_aug = lgb_aug.predict(X_test_aug)
y_pred_aug = (y_prob_aug > 0.5).astype(int)
auc_aug = roc_auc_score(y_test, y_prob_aug)

y_prob_no_age = lgb_no_age.predict(X_test_no_age)
y_pred_no_age = (y_prob_no_age > 0.5).astype(int)
auc_no_age = roc_auc_score(y_test, y_prob_no_age)

# Fairness Metrics
def get_fairness(y_true, y_pred, S):
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=S
    )

    spd = mf.difference(method="between_groups")["selection_rate"]
    eod = max(
        mf.by_group["tpr"].max() - mf.by_group["tpr"].min(),
        mf.by_group["fpr"].max() - mf.by_group["fpr"].min()
    )
    return spd, eod

spd_base, eod_base = get_fairness(y_test, y_pred_base, S_test)
spd_aug, eod_aug = get_fairness(y_test, y_pred_aug, S_test)
spd_no_age, eod_no_age = get_fairness(y_test, y_pred_no_age, S_test)

#Results
print("Baseline")
print("AUC:", round(auc_base, 4))
print("Parity:", round(spd_base, 4))
print("Equalized Odds:", round(eod_base, 4))

print("\nDemographic Inclusion")
print("AUC:", round(auc_aug, 4))
print("Parity:", round(spd_aug, 4))
print("Equalized Odds:", round(eod_aug, 4))

print("\nAge Ablation")
print("AUC:", round(auc_no_age, 4))
print("Parity:", round(spd_no_age, 4))
print("Equalized Odds:", round(eod_no_age, 4))
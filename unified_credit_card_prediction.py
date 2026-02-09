import numpy as np
import pandas as pd
import warnings, random
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings(action='ignore')

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# 1. Load Data
train_path = r'c:\Users\alsld\github\data science\pandas\신용카드 사용자 예측\train.csv'
test_path = r'c:\Users\alsld\github\data science\pandas\신용카드 사용자 예측\test.csv'
submission_path = r'c:\Users\alsld\github\data science\pandas\신용카드 사용자 예측\sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 2. Data Preprocessing
# Fill missing occupational types with 'NaN' (Notebook 1 & 2)
train.fillna('NaN', inplace=True)
test.fillna('NaN', inplace=True)

# Outlier Removal (Notebook 1)
train = train[train['family_size'] <= 7].reset_index(drop=True)

# Remove redundant/constant columns (Notebook 1)
train.drop(['index', 'FLAG_MOBIL'], axis=1, inplace=True)
test.drop(['index', 'FLAG_MOBIL'], axis=1, inplace=True)

# Handle DAYS_EMPLOYED anomalies (Notebook 1 & 2)
train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].map(lambda x: 0 if x > 0 else x)
test['DAYS_EMPLOYED'] = test['DAYS_EMPLOYED'].map(lambda x: 0 if x > 0 else x)

# Convert negative values to positive (Notebook 1)
feats = ['DAYS_BIRTH', 'begin_month', 'DAYS_EMPLOYED']
for feat in feats:
    train[feat] = np.abs(train[feat])
    test[feat] = np.abs(test[feat])

# 3. Feature Engineering
for df in [train, test]:
    # Professional life & Ratios (Notebook 1)
    df['before_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
    df['income_total_befofeEMP_ratio'] = df['income_total'] / (df['before_EMPLOYED'] + 1) # Added 1 to avoid div by zero
    
    # Time-based features (Notebook 1)
    df['before_EMPLOYED_m'] = np.floor(df['before_EMPLOYED'] / 30) - ((np.floor(df['before_EMPLOYED'] / 30) / 12).astype(int) * 12)
    df['before_EMPLOYED_w'] = np.floor(df['before_EMPLOYED'] / 7) - ((np.floor(df['before_EMPLOYED'] / 7) / 4).astype(int) * 4)
    
    df['Age'] = df['DAYS_BIRTH'] // 365
    df['DAYS_BIRTH_m'] = np.floor(df['DAYS_BIRTH'] / 30) - ((np.floor(df['DAYS_BIRTH'] / 30) / 12).astype(int) * 12)
    df['DAYS_BIRTH_w'] = np.floor(df['DAYS_BIRTH'] / 7) - ((np.floor(df['DAYS_BIRTH'] / 7) / 4).astype(int) * 4)
    
    df['EMPLOYED'] = df['DAYS_EMPLOYED'] // 365
    df['DAYS_EMPLOYED_m'] = np.floor(df['DAYS_EMPLOYED'] / 30) - ((np.floor(df['DAYS_EMPLOYED'] / 30) / 12).astype(int) * 12)
    df['DAYS_EMPLOYED_w'] = np.floor(df['DAYS_EMPLOYED'] / 7) - ((np.floor(df['DAYS_EMPLOYED'] / 7) / 4).astype(int) * 4)

    df['ability'] = df['income_total'] / (df['DAYS_BIRTH'] + df['DAYS_EMPLOYED'] + 1)
    df['income_mean'] = df['income_total'] / df['family_size']
    
    # Interaction Features (Notebook 2)
    df['income_age'] = df['income_total'] * df['Age']
    df['income_emp'] = df['income_total'] * df['EMPLOYED']
    
    # ID Feature (Notebook 1 - Critical for unique user identification)
    df['ID'] = \
    df['child_num'].astype(str) + '_' + df['income_total'].astype(str) + '_' + \
    df['DAYS_BIRTH'].astype(str) + '_' + df['DAYS_EMPLOYED'].astype(str) + '_' + \
    df['work_phone'].astype(str) + '_' + df['phone'].astype(str) + '_' + \
    df['email'].astype(str) + '_' + df['family_size'].astype(str) + '_' + \
    df['gender'].astype(str) + '_' + df['car'].astype(str) + '_' + \
    df['reality'].astype(str) + '_' + df['income_type'].astype(str) + '_' + \
    df['edu_type'].astype(str) + '_' + df['family_type'].astype(str) + '_' + \
    df['house_type'].astype(str) + '_' + df['occyp_type'].astype(str)

# Drop columns that are now redundant or show multicollinearity
cols_to_drop = ['child_num', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# 4. Encoding & Scaling
numerical_feats = train.dtypes[train.dtypes != "object"].index.tolist()
if 'credit' in numerical_feats: numerical_feats.remove('credit')
categorical_feats = train.dtypes[train.dtypes == "object"].index.tolist()

# Log scale income_total
for df in [train, test]:
    df['income_total'] = np.log1p(1 + df['income_total'])

# Ordinal Encoding for Categorical Features (Notebook 1)
encoder = OrdinalEncoder(cols=categorical_feats)
train[categorical_feats] = encoder.fit_transform(train[categorical_feats])
test[categorical_feats] = encoder.transform(test[categorical_feats])

# Convert ID to int64
train['ID'] = train['ID'].astype('int64')
test['ID'] = test['ID'].astype('int64')

# Clustering Feature (Notebook 1)
kmeans_train = train.drop(['credit'], axis=1)
# Use a subset for faster clustering setup if needed, but here we follow Notebook 1
kmeans = KMeans(n_clusters=36, random_state=seed).fit(kmeans_train)
train['cluster'] = kmeans.predict(kmeans_train)
test['cluster'] = kmeans.predict(test)

# Standard Scaling for Numerical Features (except already log-scaled income_total)
numerical_feats_to_scale = [f for f in numerical_feats if f != 'income_total']
scaler = StandardScaler()
train[numerical_feats_to_scale] = scaler.fit_transform(train[numerical_feats_to_scale])
test[numerical_feats_to_scale] = scaler.transform(test[numerical_feats_to_scale])

# 5. Modeling - CatBoost (Optimized Fold 15)
n_fold = 15
n_class = 3
target = 'credit'
X = train.drop(target, axis=1)
y = train[target]
X_test = test

skfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

cat_pred_test = np.zeros((X_test.shape[0], n_class))
cat_cols = ['income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type', 'ID']

print(f"Starting {n_fold}-fold cross validation...")

for fold, (train_idx, valid_idx) in enumerate(skfold.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)
    
    model_cat = CatBoostClassifier(
        iterations=2000,
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=False # Set to True to see progress
    )
    
    model_cat.fit(train_data, eval_set=valid_data, use_best_model=True)
    
    fold_pred = model_cat.predict_proba(X_test)
    cat_pred_test += fold_pred / n_fold
    
    print(f"Fold {fold} finished.")

# 6. Create Submission
submission = pd.read_csv(submission_path)
submission.iloc[:, 1:] = cat_pred_test
output_filename = 'unified_submission_v1.csv'
submission.to_csv(output_filename, index=False)

print(f"Submission file saved as {output_filename}")

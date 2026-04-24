# %% [code] {"execution":{"iopub.status.busy":"2026-04-24T02:01:47.076273Z","iopub.execute_input":"2026-04-24T02:01:47.077017Z","iopub.status.idle":"2026-04-24T02:01:47.451243Z","shell.execute_reply.started":"2026-04-24T02:01:47.076983Z","shell.execute_reply":"2026-04-24T02:01:47.450485Z"}}
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score

train= pd.read_parquet('/kaggle/input/competitions/ceip-ds-jecrc/train.parquet')
test= pd.read_parquet('/kaggle/input/competitions/ceip-ds-jecrc/test.parquet')
#Target fix
train['target'] = train['target'].astype(int)

# %% [code] {"execution":{"iopub.status.busy":"2026-04-24T02:01:47.452877Z","iopub.execute_input":"2026-04-24T02:01:47.453707Z","iopub.status.idle":"2026-04-24T02:01:49.610726Z","shell.execute_reply.started":"2026-04-24T02:01:47.453673Z","shell.execute_reply":"2026-04-24T02:01:49.609995Z"}}
#Features Engg.
def create_features(df):
    df = df.copy()
    
    #Date/Time
    df['Date'] = pd.to_datetime(df['Date'])
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    #Sensor Stats
    sensors = ['X1', 'X2', 'X3', 'X4', 'X5']
    df['s_mean'] = df[sensors].mean(axis=1)
    df['s_std'] = df[sensors].std(axis=1)
    df['s_max'] = df[sensors].max(axis=1)

    #Windowing
    df = df.sort_values('Date')
    for c in sensors:
        df[f'{c}_roll'] = df[c].rolling(5).mean().bfill()
        #rolling (5) add extra 0.1 to f1/Dont change optimal
        df[f'{c}_var'] = df[c].rolling(6).var().bfill()
        df[f'{c}_ewma'] = df[c].ewm(span=5, adjust=False).mean()
        df[f'{c}_diff'] = df[c].diff().fillna(0)

    df['X1_X2_ratio'] = df['X1'] / (df['X2'] + 0.001)
    df['X3_X4_ratio'] = df['X3'] / (df['X4'] + 0.001)
        
    return df

train_df= create_features(train)
test_df= create_features(test)
print(train_df.head())

# %% [code] {"execution":{"iopub.status.busy":"2026-04-24T02:01:49.611656Z","iopub.execute_input":"2026-04-24T02:01:49.611955Z","iopub.status.idle":"2026-04-24T02:02:13.133182Z","shell.execute_reply.started":"2026-04-24T02:01:49.611928Z","shell.execute_reply":"2026-04-24T02:02:13.132014Z"}}
#Drop junk columns
drops= ['Date', 'target', 'ID', 'Id', 'id']
features= [c for c in train_df.columns if c not in drops]

X = train_df[features]
y = train_df['target']

#Chronological split 73/27 (found by trial and error)
split_idx = int(len(X) * 0.73)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train.values)
X_val_s = sc.transform(X_val.values)
X_test_s = sc.transform(test_df[features].values)

neg = len(y_train[y_train == 0])
pos = len(y_train[y_train == 1])
ratio = neg / pos

print("Training LightGBM...")
lgb = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio, random_state=27, verbose=-1)
#dont change random state = 27
lgb.fit(X_train_s, y_train)

lgb_val_probs = lgb.predict_proba(X_val_s)[:, 1]

best_thresh = 0.5
best_f1 = 0

for thresh in np.arange(0.01, 1.0, 0.01):
    temp_preds = (lgb_val_probs >= thresh).astype(int)
    temp_f1 = f1_score(y_val, temp_preds, zero_division=0)
    if temp_f1 > best_f1:
        best_f1 = temp_f1
        best_thresh = thresh

print("\nIgnore Warnings")
print(f"\nF1 Score: {best_f1} at Threshold: {best_thresh}")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-24T02:02:13.134687Z","iopub.execute_input":"2026-04-24T02:02:13.135113Z","iopub.status.idle":"2026-04-24T02:02:15.888944Z","shell.execute_reply.started":"2026-04-24T02:02:13.135082Z","shell.execute_reply":"2026-04-24T02:02:15.888085Z"}}
#Submit Export
lgb_test_probs = lgb.predict_proba(X_test_s)[:, 1]
final_test_preds = (lgb_test_probs >= best_thresh).astype(int)

#submission dataframe
sub = pd.DataFrame()
if 'ID' in test.columns:
    sub['ID'] = test['ID']
elif 'Id' in test.columns:
    sub['Id'] = test['Id']
else:
    sub['Date'] = test['Date'] 
 
temp_df = test_df.copy()
temp_df['target'] = final_test_preds
temp_df = temp_df.sort_index()

sub['target'] = temp_df['target'].values
sub.to_parquet('submission.parquet', index=False)

print("\nIgnore Warnings")
print(f"\nTotal rows: {len(sub)}")
print(sub['target'].value_counts())

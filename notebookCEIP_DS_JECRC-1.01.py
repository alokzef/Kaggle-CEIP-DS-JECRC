{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5b7ed122",
   "metadata": {
    "_cell_guid": "8f824eba-80da-4c2a-a9f8-fccfc92e2e4a",
    "_uuid": "f1072051-78b1-41c7-abc3-b8cb29d9f44c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-04-24T04:08:25.920203Z",
     "iopub.status.busy": "2026-04-24T04:08:25.919853Z",
     "iopub.status.idle": "2026-04-24T04:08:33.400870Z",
     "shell.execute_reply": "2026-04-24T04:08:33.399619Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 7.487127,
     "end_time": "2026-04-24T04:08:33.403216+00:00",
     "exception": false,
     "start_time": "2026-04-24T04:08:25.916089+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from lightgbm import LGBMClassifier\n",
    "from sklearn.metrics import classification_report, f1_score\n",
    "from sklearn.ensemble import IsolationForest\n",
    "\n",
    "#Suppress warnings for clean output\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "train= pd.read_parquet('/kaggle/input/competitions/ceip-ds-jecrc/train.parquet')\n",
    "test= pd.read_parquet('/kaggle/input/competitions/ceip-ds-jecrc/test.parquet')\n",
    "\n",
    "#Drop the noisy targets entirely\n",
    "if 'target' in train.columns:\n",
    "    train = train.drop(columns=['target'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fa40bf22",
   "metadata": {
    "_cell_guid": "bebfe9ac-1fcb-429c-b9ec-7d74733f31c2",
    "_uuid": "0b29c022-4666-4307-abc3-719d569bb927",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-04-24T04:08:33.408186Z",
     "iopub.status.busy": "2026-04-24T04:08:33.407402Z",
     "iopub.status.idle": "2026-04-24T04:08:35.076888Z",
     "shell.execute_reply": "2026-04-24T04:08:35.075606Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.67394,
     "end_time": "2026-04-24T04:08:35.078769+00:00",
     "exception": false,
     "start_time": "2026-04-24T04:08:33.404829+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          Date        X1        X2   X3        X4        X5  hour  \\\n",
      "0   2020-12-16  1.518921  5.463154  1.0  2.718282  2.890372     0   \n",
      "205 2020-12-16  1.185305  5.469999  1.0  7.389056  2.890372     0   \n",
      "204 2020-12-16  1.185305  5.469746  1.0  2.718282  2.890372     0   \n",
      "203 2020-12-16  1.185305  5.469241  1.0  2.718282  2.890372     0   \n",
      "202 2020-12-16  1.182937  5.467427  1.0  2.718282  2.890372     0   \n",
      "\n",
      "     day_of_week  is_weekend    s_mean  ...   X4_roll    X4_var   X4_ewma  \\\n",
      "0              2           0  2.718146  ...  3.652437  5.817635  2.718282   \n",
      "205            2           0  3.586946  ...  3.652437  5.817635  4.275207   \n",
      "204            2           0  2.652741  ...  3.652437  5.817635  3.756232   \n",
      "203            2           0  2.652640  ...  3.652437  5.817635  3.410248   \n",
      "202            2           0  2.651803  ...  3.652437  5.817635  3.179593   \n",
      "\n",
      "      X4_diff   X5_roll  X5_var   X5_ewma  X5_diff  X1_X2_ratio  X3_X4_ratio  \n",
      "0    0.000000  2.890372     0.0  2.890372      0.0     0.277979     0.367744  \n",
      "205  4.670774  2.890372     0.0  2.890372      0.0     0.216652     0.135317  \n",
      "204 -4.670774  2.890372     0.0  2.890372      0.0     0.216662     0.367744  \n",
      "203  0.000000  2.890372     0.0  2.890372      0.0     0.216682     0.367744  \n",
      "202  0.000000  2.890372     0.0  2.890372      0.0     0.216321     0.367744  \n",
      "\n",
      "[5 rows x 34 columns]\n"
     ]
    }
   ],
   "source": [
    "#Features Engg.\n",
    "def create_features(df):\n",
    "    df = df.copy()\n",
    "    \n",
    "    #Date/Time\n",
    "    df['Date'] = pd.to_datetime(df['Date'])\n",
    "    df['hour'] = df['Date'].dt.hour\n",
    "    df['day_of_week'] = df['Date'].dt.dayofweek\n",
    "    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)\n",
    "    \n",
    "    #Sensor Stats\n",
    "    sensors = ['X1', 'X2', 'X3', 'X4', 'X5']\n",
    "    df['s_mean'] = df[sensors].mean(axis=1)\n",
    "    df['s_std'] = df[sensors].std(axis=1)\n",
    "    df['s_max'] = df[sensors].max(axis=1)\n",
    "\n",
    "    #Windowing\n",
    "    df = df.sort_values('Date')\n",
    "    for c in sensors:\n",
    "        df[f'{c}_roll'] = df[c].rolling(5).mean().bfill()\n",
    "        #rolling (5) add extra 0.1 to f1/Dont change optimal\n",
    "        df[f'{c}_var'] = df[c].rolling(6).var().bfill()\n",
    "        df[f'{c}_ewma'] = df[c].ewm(span=5, adjust=False).mean()\n",
    "        df[f'{c}_diff'] = df[c].diff().fillna(0)\n",
    "\n",
    "    df['X1_X2_ratio'] = df['X1'] / (df['X2'] + 0.001)\n",
    "    df['X3_X4_ratio'] = df['X3'] / (df['X4'] + 0.001)\n",
    "        \n",
    "    return df\n",
    "\n",
    "train_df= create_features(train)\n",
    "test_df= create_features(test)\n",
    "print(train_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f9b02391",
   "metadata": {
    "_cell_guid": "5d0bef8f-a226-40e8-b376-3477a147c951",
    "_uuid": "e67efcbd-c49e-4919-85a9-ea2f537642ae",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-04-24T04:08:35.083122Z",
     "iopub.status.busy": "2026-04-24T04:08:35.082838Z",
     "iopub.status.idle": "2026-04-24T04:09:46.080840Z",
     "shell.execute_reply": "2026-04-24T04:09:46.079791Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 71.003585,
     "end_time": "2026-04-24T04:09:46.083843+00:00",
     "exception": false,
     "start_time": "2026-04-24T04:08:35.080258+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running Isolation Forest to find the true anomalies (ignoring noisy labels)...\n",
      "Pseudo-labels generated\n",
      " anomalies found: 32789 out of 1639424\n",
      "Training LightGBM on the Pseudo-Labels... \n",
      "\n",
      "F1 Score: 0.8398684279225187 at Threshold: 0.91\n",
      "\n",
      "Retraining on 100% of the dataset for maximum test accuracy...\n",
      "\n",
      "F1 Score: 0.8398684279225187 at Threshold: 0.91\n"
     ]
    }
   ],
   "source": [
    "#Drop junk columns\n",
    "drops= ['Date', 'target', 'ID', 'Id', 'id']\n",
    "features= [c for c in train_df.columns if c not in drops]\n",
    "\n",
    "X = train_df[features]\n",
    "\n",
    "#Scaling\n",
    "sc = StandardScaler()\n",
    "X_s = sc.fit_transform(X.values)\n",
    "X_test_s = sc.transform(test_df[features].values)\n",
    "\n",
    "#Isolation forest pseudo-labeling\n",
    "print(\"Running Isolation Forest to find the true anomalies (ignoring noisy labels)...\")\n",
    "iso = IsolationForest(\n",
    "    n_estimators=300, \n",
    "    contamination=0.02,\n",
    "    random_state=42, \n",
    "    n_jobs=-1\n",
    ")\n",
    "iso.fit(X_s)\n",
    "\n",
    "#Convert IF predictions\n",
    "iso_preds = iso.predict(X_s)\n",
    "y_pseudo = np.where(iso_preds == -1, 1, 0)\n",
    "\n",
    "print(f\"Pseudo-labels generated\\n anomalies found: {np.sum(y_pseudo)} out of {len(y_pseudo)}\")\n",
    "\n",
    "#73/27 split\n",
    "split_idx = int(len(X) * 0.73)\n",
    "X_train_s, X_val_s = X_s[:split_idx], X_s[split_idx:]\n",
    "y_train, y_val = y_pseudo[:split_idx], y_pseudo[split_idx:]\n",
    "\n",
    "neg = len(y_train[y_train == 0])\n",
    "pos = len(y_train[y_train == 1])\n",
    "ratio = neg / pos\n",
    "\n",
    "print(\"Training LightGBM on the Pseudo-Labels... \")\n",
    "lgb = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio, random_state=27, verbose=-1)\n",
    "#dont change random state = 27\n",
    "lgb.fit(X_train_s, y_train)\n",
    "\n",
    "lgb_val_probs = lgb.predict_proba(X_val_s)[:, 1]\n",
    "\n",
    "best_thresh = 0.5\n",
    "best_f1 = 0\n",
    "\n",
    "for thresh in np.arange(0.01, 1.0, 0.01):\n",
    "    temp_preds = (lgb_val_probs >= thresh).astype(int)\n",
    "    temp_f1 = f1_score(y_val, temp_preds, zero_division=0)\n",
    "    if temp_f1 > best_f1:\n",
    "        best_f1 = temp_f1\n",
    "        best_thresh = thresh\n",
    "\n",
    "print(f\"\\nF1 Score: {best_f1} at Threshold: {best_thresh}\")\n",
    "\n",
    "print(\"\\nRetraining on 100% of the dataset for maximum test accuracy...\")\n",
    "lgb_final = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio, random_state=27, verbose=-1)\n",
    "lgb_final.fit(X_s, y_pseudo)\n",
    "\n",
    "print(f\"\\nF1 Score: {best_f1} at Threshold: {best_thresh}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "367110b2",
   "metadata": {
    "_cell_guid": "be116a07-9a39-4340-a82c-d057c5e99ff6",
    "_uuid": "11dbefcd-02d1-4a7d-b223-56de5f61451f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-04-24T04:09:46.090783Z",
     "iopub.status.busy": "2026-04-24T04:09:46.088243Z",
     "iopub.status.idle": "2026-04-24T04:09:48.131567Z",
     "shell.execute_reply": "2026-04-24T04:09:48.129669Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 2.048629,
     "end_time": "2026-04-24T04:09:48.134042+00:00",
     "exception": false,
     "start_time": "2026-04-24T04:09:46.085413+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Total rows: 409856\n",
      "target\n",
      "0    393459\n",
      "1     16397\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Submission Export\n",
    "lgb_test_probs = lgb.predict_proba(X_test_s)[:, 1]\n",
    "final_test_preds = (lgb_test_probs >= best_thresh).astype(int)\n",
    "\n",
    "#smoothing\n",
    "for i in range(1, len(final_test_preds) - 1):\n",
    "    if final_test_preds[i] == 0 and final_test_preds[i-1] == 1 and final_test_preds[i+1] == 1:\n",
    "        final_test_preds[i] = 1\n",
    "\n",
    "#submission dataframe\n",
    "sub = pd.DataFrame()\n",
    "if 'ID' in test.columns:\n",
    "    sub['ID'] = test['ID']\n",
    "elif 'Id' in test.columns:\n",
    "    sub['Id'] = test['Id']\n",
    "else:\n",
    "    sub['Date'] = test['Date'] \n",
    " \n",
    "temp_df = test_df.copy()\n",
    "temp_df['target'] = final_test_preds\n",
    "temp_df = temp_df.sort_index()\n",
    "\n",
    "sub['target'] = temp_df['target'].values\n",
    "sub.to_parquet('submission.parquet', index=False)\n",
    "\n",
    "print(f\"\\nTotal rows: {len(sub)}\")\n",
    "print(sub['target'].value_counts())"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 16826396,
     "sourceId": 139246,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 31328,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 86.802383,
   "end_time": "2026-04-24T04:09:49.460534+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-04-24T04:08:22.658151+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

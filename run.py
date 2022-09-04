import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the data, sample such that the target classes are equal size
df = pd.read_csv("./data/train_transaction.csv")
df = pd.concat([df[df.isFraud == 0].sample(n=len(df[df.isFraud == 1])), df[df.isFraud == 1]], axis=0)

# Select the features and target
X = df[["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "M1", "M2", "M3"]]
y = df.isFraud

# Use one-hot encoding to encode the categorical features
enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(X)

X = pd.DataFrame(enc.transform(X).toarray(), columns=enc.get_feature_names_out().reshape(-1))
X["TransactionAmt"] = df[["TransactionAmt"]].to_numpy()

# Split the dataset and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, gamma=0,
                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                    scale_pos_weight=1, seed=27)

model = xgb.fit(X_train, y_train)
# Save the model and encoder


joblib.dump(enc, "./artifacts/ohe_fraud_encoder.joblib")
joblib.dump(model, "./artifacts/xgb_fraud_model.joblib")
# preprocessing.py

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(df, target_column=None):
    df = df.copy()
    X = df.drop(columns=[target_column]) if target_column else df.copy()

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    imputer = SimpleImputer()
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    y = None
    if target_column:
        y = df[target_column]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
    return X, y

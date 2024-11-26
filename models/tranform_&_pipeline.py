import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#cargar el dataset
def load_kdd_dataset(data_path):
    """Lectura del dataSet NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns = attributes)

def train_val_test_split(df, rstate = 42, shuffle = True, stratify = None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size = 0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size = 0.5, random_state = rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)


df = load_kdd_dataset("datasets/KDD/KDDTrain+.arff")
df

train_set, val_set, test_set = train_val_test_split(df, stratify = 'protocol_type')

print("Longitud del training_set:", len(train_set))
print("logitud de validation_set:", len(val_set))
print("longitud del test_set:", len(test_set))

X_train = train_set.drop("class", axis = 1)
y_train = train_set["class"].copy()

X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"]= np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"]= np.nan
X_train


class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.dropna()


delete_nan = DeleteNanRows()
X_train_prep = delete_nan.fit_transform(X_train)
X_train_prep


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()  # Asegúrate de que este esté importado correctamente
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        
        return X_copy

custom_scaler = CustomScaler(["src_bytes", "dst_bytes"])
X_train_prep = custom_scaler.fit_transform(X_train_prep)
X_train.head(10)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('rbst_scaler', RobustScaler())
])

X_train_num = X_train.select_dtypes(exclude = ['object'])

X_train_prep = num_pipeline.fit_transform(X_train_num)
X_train_prep = pd.DataFrame(X_train_prep, columns=X_train_num.columns, index=X_train_num.index)

X_train_num.head(10)

X_train_prep.head(10)

num_attribs = list(X_train.select_dtypes(exclude = ['object']))
cat_attribs = list(X_train.select_dtypes(include = ['object']))
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

X_train_prep = full_pipeline.fit_transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns = list(pd.get_dummies(X_train)), index=X_train.index)
X_train_prep.head(10)
X_train.head(10)




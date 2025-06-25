import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df = pd.read_csv("D:\import pandas\covid_toy.csv")
df.head()
from sklearn.impute import SimpleImputer
si = SimpleImputer()
df["fever"] = si.fit_transform(df[["fever"]])
cat = df.select_dtypes(include=["object"]).columns.tolist()
for i in cat:
    df[i] = lb.fit_transform(df[i])
x = df.drop(columns=["has_covid"])
y = df["has_covid"]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x,y)
 
import joblib
joblib.dump(lr, "covid_model.pkl") 
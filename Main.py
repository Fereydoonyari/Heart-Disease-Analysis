import pandas as pd 
import zipfile
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler

zip_path = "archive.zip"
with zipfile.ZipFile(zip_path,"r") as zip_ref :
    csv_filename = zip_ref.namelist()[0] 
    with zip_ref.open(csv_filename) as file :
        df = pd.read_csv(file)

# print (df.head())
# print (df.isnull().sum())
# print (df.dtypes)
#  Identify Outliers Using Box Plots
numerical_cols = ["age","trestbps","chol","thalach","oldpeak"]
plt.figure(figsize=(12,6))
sns.boxplot(data=df[numerical_cols])
plt.title("BOX plot for outliner detection ")
plt.show()
# Points outside the whiskers are potential outliers.

def detect_outliners (df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliners = df[(df[column]<lower_bound) | (df[column] > upper_bound)]
    return outliners

for col in numerical_cols:
    outliers = detect_outliners(df,col)
    print (f"outliers in {col}: {len(outliers)}")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df [col].clip(lower_bound,upper_bound)
# data normalization 
def normalize():
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print (df.head())

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numerical_cols])
plt.title("BOX plot for outliner detection with cliping outliers ")
plt.show()
# Points outside the whiskers are potential outlie
print ("before normalization ")
print (df.head())
print ("##############")
print ("after normalization ")
normalize()

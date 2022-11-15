import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mlb
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.memmap import dtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('C:/Users/Sai Shashank/OneDrive/Desktop/churn project/Churn_Modelling.csv')
print(df.dtypes)
df.head()

print(df.isnull().sum())
df.describe()

ndf=df.copy()
ndf.head()

ndf.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
ndf.isnull().sum()

sns.boxplot(data=ndf['CreditScore'])
mlb.show()

ndf['CreditScore']=ndf['CreditScore'].fillna(df['CreditScore'].median())

sns.boxplot(data=ndf['Age'])
mlb.show

ndf['Age']=ndf['Age'].fillna(df['Age'].median())
sns.boxplot(data=ndf.Tenure)
mlb.show()
ndf['Tenure']=ndf['Tenure'].fillna(df['Tenure'].mean())

sns.boxplot(data=ndf.Balance)
mlb.show()
ndf['Balance']=ndf['Balance'].fillna(df['Balance'].median())

sns.boxplot(data=ndf.NumOfProducts)
mlb.show()
ndf['NumOfProducts']=ndf['NumOfProducts'].fillna(df['NumOfProducts'].median())

sns.boxplot(data=ndf.HasCrCard)
mlb.show()
ndf['HasCrCard']=ndf['HasCrCard'].fillna(df['HasCrCard'].mode()[0])

sns.boxplot(data=ndf.IsActiveMember)
mlb.show()
ndf['IsActiveMember']=ndf['IsActiveMember'].fillna(df['IsActiveMember'].median())

sns.boxplot(data=ndf.EstimatedSalary)
mlb.show()
ndf['EstimatedSalary']=ndf['EstimatedSalary'].fillna(df['EstimatedSalary'].mean())

ndf1=ndf.dropna(subset=['Exited'])

#ndf1=ndf1.iloc[0,0]

ndf1.isnull().sum()

ndf1.Gender.unique()

ndf1.dtypes



ndf2=ndf1.copy()
ndf2.drop(columns='Gender',inplace=True)

ndf2_with_Geo=ndf2[ndf2['Geography'].notna()]
ndf2_with_Geo.Geography.replace(['France','Spain','Germany'],['0','1','2'],inplace=True)
ndf2_with_Geo['Geography']=pd.to_numeric(ndf2_with_Geo['Geography'])
ndf2_no_Geo=ndf2[ndf2['Geography'].isna()]
X=ndf2_with_Geo.drop('Geography', axis=1).values
y=ndf2_with_Geo['Geography'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)

forest = RandomForestClassifier(n_estimators=100, random_state=100)

# Fitting a model and making predictions
forest.fit(X_train,y_train)

#predictions = forest.predict(X_test)
#print(pd.DataFrame(X_test))

import numpy as np
for i in range(len(ndf2)):
  if ndf2.iloc[i,1] not in ['Spain','France','Germany']:
      v=ndf2.iloc[i,:]
      w=v.filter(items=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
      b=np.array(w)
      b=b.reshape(1, -1)
      predictions = forest.predict(b)
      ndf2.iloc[i,1]=predictions

ndf2.Geography.replace([0,1,2],['France','Spain','Germany'],inplace=True)
ndf1['Geography']=ndf2['Geography']

#finding the Gender NaN Values
ndf2.iloc[0:0]
ndf2=ndf1.copy()
ndf2.drop(columns='Geography',inplace=True)
ndf2_with_Gender=ndf2[ndf2['Gender'].notna()]
ndf2_with_Gender.Gender.replace(['Male','Female'],['0','1'],inplace=True)
ndf2_with_Gender['Gender']=pd.to_numeric(ndf2_with_Gender['Gender'])
ndf2_no_Gender=ndf2[ndf2['Gender'].isna()]
X=ndf2_with_Gender.drop('Gender', axis=1).values
y=ndf2_with_Gender['Gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)

forest = RandomForestClassifier(n_estimators=100, random_state=100)

# Fitting a model and making predictions
forest.fit(X_train,y_train)


for i in range(len(ndf2)):
  if ndf2.iloc[i,1] not in ['Male','Female']:
      v=ndf2.iloc[i,:]
      w=v.filter(items=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
      b=np.array(w)
      b=b.reshape(1, -1)
      predictions = forest.predict(b)
      ndf2.iloc[i,1]=predictions

ndf1.Geography.unique()

ndf2.Gender.replace([0,1],['Male','Female'],inplace=True)
ndf1['Gender']=ndf2['Gender']

ndf1.Geography.unique()

ndf1.isna().sum()

ndf3=ndf1.copy()
ndf3.isna().sum()
ndf3.Geography.unique()


sns.countplot(x='Gender',hue='Exited',data=ndf3)
plt.show()

sns.countplot(x='Geography',hue='Exited',data=ndf3)
plt.show()

sns.countplot(x='IsActiveMember',hue='Exited',data=ndf3)
plt.show()

sns.countplot(x='HasCrCard',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='CreditScore',x='Exited',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='Age',x='Exited',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='Tenure',x='Exited',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='Balance',x='Exited',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='NumOfProducts',x='Exited',hue='Exited',data=ndf3)
plt.show()

sns.boxplot(y='EstimatedSalary',x='Exited',hue='Exited',data=ndf3)
plt.show()

#OneHotEncoding
ohed=pd.get_dummies(ndf3, columns = ['Geography', 'Gender'])
fdf=pd.DataFrame(ohed)
fdf.isna().sum()

fdf.to_csv('C:/Users/Sai Shashank/OneDrive/Desktop/churn project/cleaned_data.csv')
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn import svm
import streamlit as st

df=pd.read_csv('C:/Users/Sai Shashank/OneDrive/Desktop/churn project/cleaned_data.csv')
df.columns
ndf=df.copy()
ndf.drop(columns=['Unnamed: 0'],inplace=True)
ndf.isnull().sum()
ndf.columns

#univariate selection - chi2 for feature selection

x=ndf.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13]] #independent columns
y=ndf.iloc[:,8]
bestfeatures=SelectKBest(score_func=chi2,k=10)
fit=bestfeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featureScores=pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns=['Feature','Score']
print(featureScores.nlargest(10,'Score'))

#Feature Importance

model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #Feature importance is the inbuilt class of ExtraTreesClassifier
feat_importances=pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#heatmap of correlation for the continuous variables

fig,ax=mp.subplots(figsize=(15,10))
dp=sb.heatmap(ndf.corr(),cmap="YlGnBu",annot=True,linewidths=1,annot_kws={'size': 12})
mp.plot()

#removing the least significant columns from the data
z=ndf.copy()
z.drop(['Tenure','NumOfProducts','HasCrCard'],axis=1,inplace=True)
z

z.Exited.nunique()
z.shape

sb.countplot(x='Exited',data=ndf)
mp.show()

#creating training and testing data

y_data=z['Exited']
x_data=z.drop('Exited',axis=1)
x_training_data,x_test_data,y_training_data,y_test_data=train_test_split(x_data,y_data,test_size=0.2,random_state=50)

#Training Logistic Regression

model=LogisticRegression()
model.fit(x_training_data,y_training_data)
predict=model.predict(x_test_data)

#measuring the performance of model

print(classification_report(y_test_data,predict))
print(confusion_matrix(y_test_data,predict))

#OverSampling

print(f"The training dataset has {sorted(Counter(y_training_data).items())[0][1]} records for the majority class and {sorted(Counter(y_training_data).items())[1][1]} records for the minority class.")
smote=SMOTE(random_state=50)
x_train_smote,y_train_smote= smote.fit_resample(x_training_data,y_training_data)
print(sorted(Counter(y_train_smote).items()))

v=pd.DataFrame(y_train_smote)
sb.countplot(x='Exited',data=v)
mp.show()

model=LogisticRegression()
model.fit(x_train_smote,y_train_smote)
predict=model.predict(x_test_data)

print(classification_report(y_test_data,predict))
print(confusion_matrix(y_test_data,predict))

#Under-Sampling Using NearMiss

nearmiss = NearMiss(version=3)
x_train_nearmiss,y_train_nearmiss= nearmiss.fit_resample(x_training_data,y_training_data)
# Check the number of records after over sampling
print(sorted(Counter(y_train_nearmiss).items()))

v=pd.DataFrame(y_train_nearmiss)
sb.countplot(x='Exited',data=v)
mp.show()

model=LogisticRegression()
model.fit(x_train_nearmiss,y_train_nearmiss)
predict=model.predict(x_test_data)

print(classification_report(y_test_data,predict))
print(confusion_matrix(y_test_data,predict))
st.title("Employee Churn prediction")
user_input_credit_score=st.number_input("Enter credit score", key=1)
user_input_age=st.number_input("Enter age",key=2)
user_input_balance=st.number_input("Enter balance",key=3)
user_input_isactive=st.number_input("Enter isActive",key=4)
user_input_estimatedsalary=st.number_input("Enter estimated salary",key=5)
user_input_geography=st.text_input("Enter geography",key=6)
user_input_gender= st.text_input("Enter Gender")

idf = pd.DataFrame()
idf['CreditScore'] = [float(int(user_input_credit_score))]
idf['Age'] = [float(user_input_age)]
idf['Balance'] = [float(user_input_balance)]
idf['IsActiveMember'] = [float(user_input_isactive)]
idf['EstimatedSalary'] = [float(user_input_estimatedsalary)]
geo = user_input_geography
if geo == 'France':
  idf['Geography_France'] = [1]
  idf['Geography_Germany'] = [0]
  idf['Geography_Spain'] = [0]
elif geo == 'Germany':
  idf['Geography_France'] = [0]
  idf['Geography_Germany'] = [1]
  idf['Geography_Spain'] = [0]
elif geo == 'Spain':
  idf['Geography_France'] = [0]
  idf['Geography_Germany'] = [0]
  idf['Geography_Spain'] = [1]
gen = user_input_gender
if gen == 'Male':
  idf['Gender_FeMale'] = [0]
  idf['Gender_Male'] = [1]
elif gen == 'Female':
  idf['Gender_Female'] = [1]
  idf['Gender_Male'] = [0]

idf.head()
b = np.array(idf)
b = b.reshape(1, -1)
submit=st.button("Predict")
if submit:
  predict = model.predict(b)
  if predict[0] == 0.:
    print("employee will stay")
    st.text("Customer will stay")
  else:
    print("employee will not stay")
    st.text("Customer will not stay")





#svm


# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
#
# #Train the model using the training sets
# clf.fit(x_training_data,y_training_data)
#
# #Predict the response for test dataset
# y_pred = clf.predict(x_test_data)
#
# print(classification_report(y_test_data,predict))
# print(confusion_matrix(y_test_data,predict))




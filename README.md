# cdac_project
Vehicle Accident predication model based on weather data
#importing numpy and pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""**Import Dataset**"""

first_dataset=pd.read_csv("Train_WeatherData1.csv")

first_dataset.shape

second_dataset=pd.read_csv("Train_Vehicletravellingdata1.csv")

second_dataset.shape

df=pd.concat([first_dataset, second_dataset],axis=1)
df.head()

df.to_csv('project2.csv')

df5=pd.read_csv("project2.csv",index_col=0,na_values=' ')
df5.head()

"""shape of dataset 

"""

df5.shape

"""## Datatype of dataset"""

df5.dtypes

"""Convert datatype of "DateTime" cloumn string object into datetime object"""

df5["DateTime"]= pd.to_datetime(df5["DateTime"])

df5.dtypes

"""**Check isNull value is present**"""

df5.isnull().sum()

"""**Replace all Null values with mode**"""

def fill_with_mode(x):
  for col in x:
    df5[col].fillna(df5[col].mode()[0],inplace=True)
col=['time_gaprec','weather_intensity','humidity','wind_direc','wind_speed']
fill_with_mode(col)

"""After Replacing all Null values checking is there Null values still present"""

df5.isnull().sum(axis=0)

"""**Label mapping**"""

df5['intensity'].unique()

label_map = {'intensity' : {'None' : 1, 'Low' : 2 , 'Moderate' : 3,'High' : 4}}
df5= df5.replace(label_map)

df5['intensity'].unique()

"""Fill Null value  with mode"""

def fill_with_mode(x):
  for col in x:
    df5[col].fillna(df5[col].mode()[0],inplace=True)
col=['intensity']
fill_with_mode(col)

df5['intensity'].unique()

df5.isna().sum()

df5.shape

"""**drop columns**"""

df5.drop(['ID','ID.1','DateTime','ID_prec'],axis=1,inplace=True)

df5.dtypes

"""# **Visualization**"""

import seaborn as sns
plt.figure(figsize=(6,4))
sns.barplot('condition_of_road','lane_road_accident',hue='condition_of_road',data=df5,ci=None, palette='Set2')
plt.legend(bbox_to_anchor=(1,1))
plt.title('lane_road_accident by condition_of_road')

"""**The above graph is the 'condition_of_road' with repsect to 'road_accident'.**
**By considering 'Snow_covered' condition the chances of accident is high as compared others.**
"""

import seaborn as sns
plt.figure(figsize=(10,4))
sns.barplot('percep_type','lane_road_accident',hue='percep_type',data=df5,ci=None, palette='Set2')
plt.legend(bbox_to_anchor=(1,1))
plt.title('percep_type by lane_road_accident')

"""**Observing above graph it is state that during percep_type weather i.e when Weather is clear the chances of getting accident is high as compared others and in 'snow','Rain' condition chances is low**"""

import seaborn as sns
plt.figure(figsize=(6,4))
sns.barplot('intensity','lane_road_accident',hue='intensity',data=df5,ci=None, palette='Set2')
plt.legend(bbox_to_anchor=(1,1))
plt.title('Intensity by lane_road_accident')

"""**By observing above graph the chances of getting accident due to weather intensity 'None' (i.e when there is no intensity is present)is high as comparing other condition**"""

import seaborn as sns
plt.figure(figsize=(6,4))
sns.barplot('light_condition','lane_road_accident',hue='light_condition',data=df5,ci=None, palette='Set2')
plt.legend(bbox_to_anchor=(1,1))
plt.title('lane_road_accident by light_condition')

"""**The above graph is light_condition vs road_accident.By observing the chances of getting accident is high during night and in daylight there is low chances of getting accident**"""

fig,ax=plt.subplots(1,2,figsize=(20,6))
sns.countplot('wind_speed',data=df5,hue='lane_road_accident',palette='autumn',ax=ax.flat[0])
(df5['lane_road_accident'].value_counts(normalize=True)*100).plot(kind='pie',ax=ax.flat[1])
plt.show()

"""**Label Encoder**"""

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
var = ['light_condition','percep_type','condition_of_road']
for item in var:
  df5[item] = lb.fit_transform(df5[item])

"""check duplicates value is present or not?

"""

df5.drop_duplicates().shape

"""**Histogram Plotting**"""

plt.figure(figsize=(15,25))

for i, col in enumerate(df5.columns,1):
    plt.subplot(12,2, i)
    df5[col].hist()
    plt.xlabel(col)
plt.tight_layout()

df5.head()

df5.shape

df5.describe()

"""**Box Plot**"""

plt.figure(figsize=(15,26))

for i, col in enumerate(df5.columns,1):
    plt.subplot(12,2, i)
    plt.boxplot(df5[col])
    plt.xlabel(col)
plt.tight_layout()

"""boxplot of individual features 'weight' """

df5[['weight']].boxplot()

df5['weight'].describe()

#df5['weight']= np.log(df5['weight'])

df5['weight'].describe()

df5[['leng_prec']].boxplot()

df5['leng_prec'].describe()

#df5['leng_prec']= np.log(df5['leng_prec'])

df5['leng_prec'].describe()

df5[['time_gaprec']].boxplot()

df5['time_gaprec'].describe()

#df5['time_gaprec']= np.log(df5['time_gaprec'])

df5['time_gaprec'].describe()

df5[['time_gaprec']].boxplot()

df5.columns

"""**Normalization**"""

#normalization

def normalize(x):
  return (x-np.min(x))/(max(x)-min(x))

df5=df5.apply(normalize)

df5.describe()

"""**Split Train Test Dataset**

Considering road_accident Feature for predication
"""

Y = df5['lane_road_accident']
X = df5.drop('lane_road_accident', axis = 1)

X

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size  = 0.25, random_state =7)

x_train.shape,x_test.shape

"""**Correlation of x_train**

**heatmap**
"""

import seaborn as sns
plt.figure(figsize=(15,15))
cor=x_train.corr()
sns.heatmap(cor, annot=True,cmap=plt.cm.CMRmap_r)
plt.show()

def corr(x,threshold):
  col_corr=set()
  corr_matrix=x.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if (corr_matrix.iloc[i,j]) >threshold:
        colname=corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr

corr_features=corr(x_train,0.7)
len(set((corr_features)))

corr_features

x_train.shape,x_test.shape

"""**Applying Algorithms**

**Logistic Regression**
"""

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
log_model = LogisticRegression()
#fit the model
log_model.fit(x_train, y_train)
y_t1=log_model.predict(x_test)
y_t2=log_model.predict(x_train)
#evaluate model
score=accuracy_score(y_test,y_t1)
score1=accuracy_score(y_train,y_t2)
print(score)
print(score1)

"""**ROC CURVE**"""

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, y_t1, pos_label=1)
#fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

import matplotlib.pyplot as plt


# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show();

"""**After Tunning**"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

solvers=['newton-cg','lbfgs','liblinear']
penalty=['l2']
c_values=[100,10,1.0,0.1,0.01,1000]
grid=dict(solver=solvers,penalty=penalty,C=c_values)
model_params={
    'logistic':{
        'model':LogisticRegression(),
        'params':grid
    }
    
}

scores=[]

for model_name,mp in model_params.items():
  clf1=GridSearchCV(mp['model'],mp['params'],cv=10,return_train_score=False)
  clf1.fit(x_test,y_test)
  scores.append({
      'model':model_name,
      'best_score':clf1.best_score_,
      'best_params':clf1.best_params_
  })
logistic_df=pd.DataFrame(scores,columns=['model','best_score','best_params'])
logistic_df

"""Accuracy2=63.47%

# Ridge Classifier
"""

from sklearn.linear_model import RidgeClassifier
rc = RidgeClassifier(alpha=1, normalize=True)
rc.fit(x_train, y_train)
rc_pred = rc.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
import sklearn.metrics as metrics
print(confusion_matrix(y_test,rc_pred))
print(metrics.accuracy_score(y_test,rc_pred))
print(metrics.recall_score(y_test,rc_pred))
print(metrics.precision_score(y_test,rc_pred))
print(metrics.f1_score(y_test,rc_pred))

from sklearn.linear_model import RidgeClassifier
rc = RidgeClassifier(alpha=1,normalize=True)
rc.fit(x_train, y_train)

param_rc={
    
    'max_iter':range(5,20),
    'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5],
    'normalize':[True,False],
    #'tol':[1e-2,1e-4,1e-6,1e-8,1e-10]
    'solver':['svd','sparse_cg']

    
    }
rc_grid=GridSearchCV(estimator=rc,param_grid=param_rc,cv=5,verbose=1)
rc_grid.fit(x_train,y_train)

from sklearn.linear_model import RidgeClassifier
rct = RidgeClassifier(alpha=1,max_iter=20,solver='sparse_cg',random_state=9)
rct.fit(x_train, y_train)
rct_pred = rct.predict(x_test)
print(confusion_matrix(y_test,rct_pred))
print(metrics.accuracy_score(y_test,rct_pred))
print(metrics.recall_score(y_test,rct_pred))
print(metrics.precision_score(y_test,rct_pred))
print(metrics.f1_score(y_test,rct_pred))

"""# **Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier

dt_default=DecisionTreeClassifier(criterion='entropy',random_state=3)
dt_default.fit(x_train,y_train)

dt_default_pred=dt_default.predict(x_test)

confusion_matrix(y_test,dt_default_pred)

metrics.accuracy_score(y_test,dt_default_pred)

"""Hyper tunning"""

from sklearn.tree import DecisionTreeClassifier


# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": range(10,25,5),
              "min_samples_split": range(50,200,25),
              "min_samples_leaf": range(50,200,25),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
dtree = DecisionTreeClassifier(random_state=7)


dtree_cv = GridSearchCV(dtree, param_dist, cv=5,verbose=1
                        ,return_train_score=True)

# Fit it to the data
dtree_cv.fit(x_train,y_train)

dtree_scores=pd.DataFrame(dtree_cv.cv_results_)
dtree_scores

print("best_accuracy",dtree_cv.best_score_)
print(dtree_cv.best_estimator_)
print(dtree_cv.best_params_)

dtree_tn=DecisionTreeClassifier(criterion='entropy',
                                random_state=7,
                                max_depth=25,#15
                                min_samples_leaf=125,#95
                                min_samples_split=60)#50

dtree_tn.fit(x_train,y_train)

dtree_tn.score(x_test,y_test)

"""
accuracy  is 67.23%"""

plt.figure()
plt.plot(dtree_scores["param_max_depth"],
         dtree_scores["mean_train_score"],
         label="training accuracy")

plt.plot(dtree_scores["param_max_depth"],
         dtree_scores["mean_test_score"],
         label="testing accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

"""# **RANDOM FOREST**"""

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=1,criterion='entropy',random_state=7)
rfc.fit(x_train,y_train)

rfc_pred=rfc.predict(x_test)

print(confusion_matrix(y_test,rfc_pred))
print(metrics.accuracy_score(y_test,rfc_pred))

"""Hyperparamter Tunning"""

param_grid = { 
    'n_estimators': range(25,100,25),
    'max_depth' : range(1,25,5),
    'criterion' :['gini','entropy']
}

CV_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5, verbose=1)

CV_rfc.fit(x_train, y_train)

print(CV_rfc.best_estimator_)
print(CV_rfc.best_score_)

rfc_t=RandomForestClassifier(criterion='entropy',
                                random_state=7,
                                max_depth=21,
                             n_estimators=75)
rfc_t.fit(x_train,y_train)

y_pred_rfc=rfc_t.predict(x_test)

print(confusion_matrix(y_test,y_pred_rfc))
print(metrics.accuracy_score(y_test,y_pred_rfc))

"""Accuracy after tunning 70.32"""

import sklearn.metrics as metrics
print(metrics.recall_score(y_test,y_pred_rfc))
print(metrics.precision_score(y_test,y_pred_rfc))
print(metrics.f1_score(y_test,y_pred_rfc))

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred_rfc, pos_label=1)
#fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

import matplotlib.pyplot as plt


# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Random Forest')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show();

"""# **ADABOOST CLASSIFIER**"""

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=7)
ada.fit(x_train, y_train)
ada_pred = ada.predict(x_test)

print(confusion_matrix(y_test,ada_pred))
print(accuracy_score(y_test,ada_pred))

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ada, x_train, y_train, cv=5)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(ada.score(x_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, ada_pred)))

"""# **xgboost**"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier(n_estimators=100,random_state=7)
model.fit(x_train,y_train)

model_pred=model.predict(x_test)

print(confusion_matrix(y_test,model_pred))
print(accuracy_score(y_test,model_pred))

"""Before tuning Accuracy 69.09%

**After tuning**
"""

param_grid = { 
    'base_score':[0.8,0.9,1.0,2.0,0.5],
    'learning_rate':[0.1,0.2,0.3,0.4],
    'gamma' :[1]
}

xgboost_df = GridSearchCV(estimator=model,param_grid=param_grid,cv=8, verbose=1)

xgboost_df.fit(x_train, y_train)

print(xgboost_df.best_estimator_)
print(xgboost_df.best_score_)

model_pred2=xgboost_df.predict(x_test)

print(confusion_matrix(y_test,model_pred2))
print(metrics.accuracy_score(y_test,model_pred2))

"""After Tuning Accuracy is 70.24%

**Plotting roc curve of random forest and xgboost**
"""

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, model_pred, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred_rfc, pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

import matplotlib.pyplot as plt


# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='XGBOOST')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='RF')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show();

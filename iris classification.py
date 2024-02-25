#!/usr/bin/env python
# coding: utf-8

# # pattern recognition project #2 Fatemeh Parsa

# ## Part One iris dataset

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import discriminant_analysis
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


## data load and split 


# In[3]:


file_path = "iris.data"
data = pd.read_csv(file_path, delimiter=",",names=[0,1,2,3,4])
y_n = data[data.columns[4]]
y= LabelEncoder().fit_transform(y_n) # convert y from nominal to numerical
x = data[data.columns[0:4]]
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.3 ,random_state=42 )#پارامتر ها براي تست عوض مي شود
# 2_fold cross validation
k=1# پارامتر براي تغيير فولد عوض مي شود
if k==1 :
    x_val, x_train , y_val, y_train = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
else:
    x_train, x_val, y_train , y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
x_train=x_train.reset_index(drop=True)
x_val=x_val.reset_index(drop=True)


# In[4]:


#LDA


# In[5]:



lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
X_trafo_sk = lda.fit_transform(x,y)#پارامتر ها براي تست و وليديشن عوض مي شود
X_trafo_sk =pd.DataFrame(X_trafo_sk)
y_h=lda.predict(x)#پارامتر ها براي تست و وليديشن عوض مي شود
X_trafo_sk = lda.fit_transform(x,y)###پارامتر ها براي تست و وليديشن عوض مي شود
X_trafo_sk =pd.DataFrame(X_trafo_sk)###پارامتر ها براي تست و وليديشن عوض مي شود
#ploting
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_figheight(4)
fig.set_figwidth(10)
ax1.scatter(X_trafo_sk[0],X_trafo_sk[1], c=y, cmap='cool')#پارامتر ها براي تست و وليديشن عوض مي شود
ax1.set_title('original labels _ valset _ all attributes',color='#fff')
ax2.scatter(X_trafo_sk[0],X_trafo_sk[1], c=y_h, cmap='cool')
ax2.set_title('predicted labels _ valset _ all attributes',color='#fff')
plt.tight_layout()
plt.show()
from sklearn.metrics import plot_confusion_matrix
print("confusion matrix is:")
import seaborn as sn
df_cm = pd.DataFrame(confusion_matrix(y,y_h))#پارامتر ها براي تست و وليديشن عوض مي شود
plt.figure(figsize = (3,3))
sn.heatmap(df_cm, annot=True ,cmap='cool' )


# In[6]:


# MLE


# In[7]:


### calculation of covvariance matrix and mean of each class


# In[8]:


y_train = pd.DataFrame(y_train, columns=['class'])
general = pd.concat([x_train,y_train], axis=1)
general_mean = x_train.mean()
class_means = general.groupby('class').mean()
print(class_means, '\n')
xc0 = general[general['class'] == 0]  
xc1 = general[general['class'] == 1]  
xc2 = general[general['class'] == 2] 
xc0_0 = xc0[xc0.columns[0]]
xc0_1 = xc0[xc0.columns[1]]
xc0_2 = xc0[xc0.columns[2]]
xc0_3 = xc0[xc0.columns[3]]
xc0 = np.array([xc0_0,xc0_1,xc0_2,xc0_3])
covMatrix_c0 = np.cov(xc0,bias=True)
xc1_0 = xc1[xc1.columns[0]]
xc1_1 = xc1[xc1.columns[1]]
xc1_2 = xc1[xc1.columns[2]]
xc1_3 = xc1[xc1.columns[3]]
xc1 = np.array([xc1_0,xc1_1,xc1_2,xc1_3])
covMatrix_c1 = np.cov(xc1,bias=True)
xc2_0 = xc2[xc2.columns[0]]
xc2_1 = xc2[xc2.columns[1]]
xc2_2 = xc2[xc2.columns[2]]
xc2_3 = xc2[xc2.columns[3]]
xc2 = np.array([xc2_0,xc2_1,xc2_2,xc2_3])
covMatrix_c2 = np.cov(xc2,bias=True)
print ('cov class 0', '\n' ,covMatrix_c0,'\n' ,'cov class 1','\n',
       covMatrix_c1 , '\n' ,'cov class 2', '\n' ,covMatrix_c2)


# In[9]:


import math
X=np.matrix([[1],[1],[1],[1]])
mean_c=np.array([np.matrix( class_means.loc[0]).T,
                 np.matrix( class_means.loc[1]).T,
                 np.matrix( class_means.loc[2]).T])
covMatrix_c=np.array([np.matrix(covMatrix_c0),
                     np.matrix(covMatrix_c1),
                     np.matrix(covMatrix_c2)])
c=[0,1,2]
gx=[0,1,2]
y_prediction = y_val # فقط براي اينكه هم بعد باشند با آرايه ليبل اصلي مقدار دهي اوليه كرده ام. ولي در ادامه پرديكشن مقدار جديد ميگيرد 
for j in range(52):
    X=np.matrix(x_val.loc[j]).T
#    X=np.matrix([[1],[1],[1],[1]])

    for i in range(3):
        c[i]=((-1/2)*np.matrix(mean_c[i]).T*np.linalg.inv(covMatrix_c[i])*np.matrix(mean_c[i]))-((-1/2)*math.log(np.linalg.det(np.linalg.inv(covMatrix_c[i]))))
        gx[i]=(X.T*((-1/2)*np.linalg.inv(covMatrix_c[i]))*X)+((np.linalg.inv(covMatrix_c[i])*np.matrix(mean_c[i])).T*X)+c[i]
        gx###هر سمپل متعلق به كلاسي است كه مقدارش در اين آرايه بيشتر باشد .
    a = np.array(gx)
    y_prediction[j] = np.argmax(a)#اينجا پرديكشن مقدار مي گيرد. 
    
from sklearn.metrics import plot_confusion_matrix
print("confusion matrix is:")
import seaborn as sn
df_cm = pd.DataFrame(confusion_matrix(y_val,y_prediction))#پارامتر ها براي تست و وليديشن عوض مي شود
plt.figure(figsize = (3,3))
sn.heatmap(df_cm, annot=True ,cmap='cool' )


# In[ ]:





# In[ ]:





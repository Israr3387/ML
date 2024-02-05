#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df=pd.read_csv('Advertising.csv')


# In[9]:


df.head(3)


# In[10]:


df.isna().sum()


# In[5]:


#EDA
sns.scatterplot(df['TV'],df['sales'])


# In[11]:


fig, axs = plt.subplots(1, 3)   #(rows,cols)
df.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16,6))#width and height
df.plot(kind='scatter', x='radio', y='sales', ax=axs[1])
df.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2])


# In[12]:


#correlation b/w the features
df.corr()


# In[13]:


#TV and sales are linear to each other
#Check the correlation b/w the feature
sns.heatmap(df.corr(), annot = True);


# In[14]:


#Here feature TV is more correlated with the dependent variable sales
#feature_cols = ['TV']
#x = df[feature_cols] # Features
#y = df.sales # Target variable
x=df[['TV']]
y=df[['sales']]
#Here we are selecting the whole column not only data that's why we are using df[['TV']]


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.25, random_state = 42)


# In[17]:


#Reshape is required to fit the model
X_train.shape


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


model=LinearRegression()


# In[20]:


model.fit(X_train,y_train)


# In[21]:


model.intercept_


# In[22]:


y_pred=model.predict(X_test)


# In[23]:


#Metrics for Linear Regression


# In[24]:


from sklearn import metrics


# In[25]:


model.score(X_train,y_train)#bydefault r2_score


# In[26]:


model.score(X_test,y_test)


# In[27]:


#model accuracy for testing data
metrics.r2_score(y_test,y_pred)


# In[28]:


#model accuracy for training data
metrics.r2_score(y_train,model.predict(X_train))


# In[42]:


#metrics for training data
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_train,model.predict(X_train))
mse=mean_squared_error(y_train,model.predict(X_train))
print("absolute mean error:",mae)
print("mean squared error:",mse)
print("r2 error:",metrics.r2_score(y_train,model.predict(X_train)))


# In[61]:


# metrics for testing data
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_test,model.predict(X_test))
mse=mean_squared_error(y_test,model.predict(X_test))
print("absolute mean error:",mae)
print("mean squared error:",mse)
print("r2 error:",metrics.r2_score(y_test,model.predict(X_test)))


# In[46]:


#Visualization the testset result
plt.scatter(X_test,y_test,color='b')
plt.plot(X_train,model.predict(X_train),color='g')
plt.xlabel('TV')
plt.ylabel('sales')
plt.title('TV Vs Sales testset chart')


# In[47]:


#Visualization the trainset result
plt.scatter(X_train,y_train,color='b')
plt.plot(X_test,model.predict(X_test),color='g')
plt.xlabel('TV')
plt.ylabel('sales')
plt.title('TV Vs Sales for trainset')
plt.show()


# # MultiLinear regression

# In[48]:


mul_model=LinearRegression()


# In[49]:


df.columns


# In[50]:


mul_X=df[['TV','radio', 'newspaper']]


# In[51]:


mul_y=df[['sales']]


# In[52]:


mul_X_train,mul_X_test,mul_y_train,mul_y_test=train_test_split(mul_X,mul_y,test_size=.25,random_state=42)


# In[53]:


mul_model.fit(mul_X_train,mul_y_train)


# In[54]:


mul_y_test=mul_y_test.to_numpy()


# In[55]:


mul_model.predict(mul_X_test)


# In[56]:


mul_y_pred=mul_model.predict(mul_X_test)                                                         


# In[57]:


mul_y_pred.ndim


# In[58]:


mul_y_test.ndim


# In[59]:


mul_model.score(mul_X_train,mul_y_train)


# In[60]:


mul_model.score(mul_X_test,mul_y_test)


# # Metrics for multiLinear regression 

# In[68]:


#metrics for training data
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(mul_y_train,mul_model.predict(mul_X_train))
mae=mean_absolute_error(mul_y_train,mul_model.predict(mul_X_train))
r2=r2_score(mul_y_train,mul_model.predict(mul_X_train))
rmse=np.sqrt(mse)
print(f"mse:{mse},mae:{mae},r2:{r2},rmse:{rmse}")


# In[69]:


#metrics for testing data
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(mul_y_test,mul_model.predict(mul_X_test))
mae=mean_absolute_error(mul_y_test,mul_model.predict(mul_X_test))
r2=r2_score(mul_y_test,mul_model.predict(mul_X_test))
rmse=np.sqrt(mse)
print(f"mse:{mse},mae:{mae},r2:{r2},rmse:{rmse}")


# In[ ]:





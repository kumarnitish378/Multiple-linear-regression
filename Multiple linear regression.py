#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
from sklearn import linear_model
import math


# In[32]:


df = pd.read_csv('home.csv')
df


# In[33]:


df = pd.read_csv('home.csv')
df.bedroom = df.bedroom.fillna(math.floor(df.bedroom.median())) # fill NaN data with mean of number of bedroom
df


# In[34]:


# equation of multiple linear regression is,[ y=m1*x1 + m2*x2 + m3*x3 + ...Mn*Xn + c ]

reg = linear_model.LinearRegression() #creat object, load  regression module for calculation
reg.fit(df[['area','bedroom','age']],df.price) # training data, here [ area = x1, bedroom = x2, age = x3 ] and price = y


# In[35]:


reg.coef_   # printint value of m1,m2,m3, (where m1,m2,m3 is slop)


# In[36]:


reg.intercept_ # printing value of c, where c is intercept point,


# In[37]:


reg.predict(df[['area','bedroom','age']]) # prediction of data


# In[38]:


y_pred = reg.predict([[5000,5,10]]) # prediction of data
y_pred

# we can also predict price by using above formula with the help of value of m1,m2,m3 and c
# [ y=m1*x1 + m2*x2 + m3*x3 + ...Mn*Xn + c ] formula


# In[ ]:


#continuos prediction of price:
print('_____________________________________________________')
while True:
    area = float(input("enter area of plot >> "))
    bedroom = float(input("enter number of bedroom >> "))
    age = float(input("enter age of plot >> "))
    price = reg.predict([[area,bedroom,age]])
    if price <0:
        print("enter vailed data")
    elif price>0:
        print("price of this plot is {}".format(price[[0]]))
    if input("you want to continue [Y/N]? ") == "n":
        break
    print('_____________________________________________________')


# In[ ]:





# In[ ]:





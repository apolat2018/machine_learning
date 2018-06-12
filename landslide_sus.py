# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:07:42 2018

@author: ap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
#Verilerin hazırlanması işlemi-------------------------------------------------
from sklearn.preprocessing import Imputer#kayıp değerleri düzenlemek için 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#eğitim verileri okunuyor
veriler=pd.read_excel("train.xlsx")
#analiz edeceğimiz veriler okunuyor
analiz=pd.read_excel("predict.xlsx")
koor=pd.read_excel("koordinatlar.xlsx").values

##eğitim verileri bağımlı ve bağımsız değişkenler olarak ayrılıyor.

##parametreler bağımsız değişkenler
parametreler=veriler.iloc[:,1:9].values
#cls (sınıf) bağımlı değişken
cls=veriler.iloc[:,:1].values

s_train=veriler.shape[0]
s_analiz=analiz.shape[0]
koor=pd.DataFrame(data=koor,index=range(s_analiz),columns=["x","y"])
imputer= Imputer(missing_values='NaN', strategy = 'median', axis=0 )
parametreler=imputer.fit_transform(parametreler)
#analiz verilerindeki kayıp veriler de de ortalamalar alındı
analiz=imputer.fit_transform(analiz)  

x=pd.DataFrame(data=parametreler,index=range(s_train),columns=["asp","crv","elevation","lito","ridge","slope","TWI","road"])
x=x.drop(columns=["road"])
pre=pd.DataFrame(data=analiz,index=range(s_analiz),columns=["asp","crv","elevation","lito","ridge","slope","TWI","road"])
pre=pre.drop(columns=["road"])
y=pd.DataFrame(data=cls,index=range(s_train),columns=["class"])


def linear_reg(x,y,predict):
    lin_reg=LinearRegression()
    lin_reg.fit(x,y)
    model=sm.OLS(lin_reg.predict(x),x)
    print(model.fit().summary())
    print("Linear Regression R2 Değeri:")
    print (r2_score(y,lin_reg.predict(x)))
    tahmin=lin_reg.predict(predict)
    lin_r=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["lin_r"])
    lin_r=pd.concat([koor,lin_r],axis=1)
    lin_r.to_excel("linreg.xlsx")
    
    print ("Linear Regression işlemi bitti")

#linear_reg(x,y,pre)

def polynomial(x,y,predict):
    poly_reg=PolynomialFeatures(degree=3)
    x_poly=poly_reg.fit_transform(x)
    lin_reg=LinearRegression()
    lin_reg.fit(x_poly,y)
    model=sm.OLS(lin_reg.predict(poly_reg.fit_transform(x)),x)
    print(model.fit().summary())
    print (r2_score(y,lin_reg.predict(poly_reg.fit_transform(x))))
    tahmin=lin_reg.predict(poly_reg.fit_transform(predict))
    lin_r=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["lin_r"])
    poly_son=pd.concat([koor,lin_r],axis=1)
    poly_son.to_excel("linreg_poly.xlsx")
    print ("Polynomial Regression işlemi bitti")

#polynomial(x,y,pre)

def sv_reg (x,y,predict):
    sc1=StandardScaler()
    x_olcekli=sc1.fit_transform(x)
    svr_reg=SVR(kernel="rbf")
    svr_reg.fit(x_olcekli,y)
    model=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
    print(model.fit().summary())
    print (r2_score(y,svr_reg.predict(x_olcekli)))
    tahmin=svr_reg.predict(predict)
    svr_r=pd.DataFrame(data=tahmin,columns=["svr_r"])
    svr_r.to_excel("SVR.xlsx")
    print ("Support Vector Regression işlemi bitti")
    
sv_reg(x,y,pre)

def dt_reg(x,y,predict):
    dt=DecisionTreeRegressor(random_state=0)
    dt.fit(x,y)
    model=sm.OLS(dt.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,dt.predict(x)))
    tahmin=dt.predict(predict)
    dt_r=pd.DataFrame(data=tahmin,columns=["dt_r"])
    dt_r.to_excel("dt_r.xlsx")
    print ("Decision tree Regression işlemi bitti")

#dt_reg(x_son,y,pre_son)
    
def rf_reg(x,y,predict):
    rf=RandomForestRegressor(n_estimators=10,random_state=0)
    rf.fit(x,y)
    model=sm.OLS(rf.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,rf.predict(x)))
    tahmin=rf.predict(predict)
    rf_r=pd.DataFrame(data=tahmin,columns=["rf_r"])
    rf_r.to_excel("rf_r.xlsx")
    print ("Random Forest Regression işlemi bitti")

#rf_reg(x_son,y,pre_son)
def logistic_regression_cls(x,y,predict):
    log_r=LogisticRegression(random_state=0)
    log_r.fit(x,y)
    
    model=sm.OLS(log_r.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,log_r.predict(x)))
    tahmin=log_r.predict_proba(predict)
    lr=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["Sifir","Bir"])
    lr=pd.concat([koor,lr],axis=1)
   
    lr.to_excel("logistic_r.xlsx",columns=["x","y","Bir"])
    print ("Logistic regresyon işlemi bitti")

    
#logistic_regression_cls(x,y,pre)

def KNN_cls(x,y,predict):
    knn=KNeighborsClassifier(n_neighbors=11,metric="minkowski")
    knn.fit(x,y)
    
    model=sm.OLS(knn.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,knn.predict(x)))
    tahmin=knn.predict_proba(predict)
    knn_cls=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["Sifir","Bir"])
    knn=pd.concat([koor,knn_cls],axis=1)
   
    knn.to_excel("knn.xlsx",columns=["x","y","Bir"])
   
  
    print ("KNN işlemi bitti")

#KNN_cls(x,y,pre)

def naive_bayes_cls(x,y,predict):
    nb=GaussianNB()
    nb.fit(x,y)
    
    model=sm.OLS(nb.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,nb.predict(x)))
    tahmin=nb.predict_proba(predict)
    nb_cls=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["Sifir","Bir"])
    nb=pd.concat([koor,nb_cls],axis=1)
   
    nb.to_excel("nb.xlsx",columns=["x","y","Bir"])
   
    print ("Naive Bayes işlemi bitti")
    
#naive_bayes_cls(x,y,pre)

def SVM_cls(x,y,predict):
    svm=SVC(kernel="rbf",probability=True)
    svm.fit(x,y)
    
    model=sm.OLS(svm.predict(x),x)
    print(model.fit().summary())
    print ("R2 değeri:",r2_score(y,svm.predict(x)))
    tahmin=svm.predict_proba(predict)
    svm_cls=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["Sifir","Bir"])
    svm_cls=pd.concat([koor,svm_cls],axis=1)
   
    svm_cls.to_excel("svm.xlsx",columns=["x","y","Bir"])
   
    
    
#SVM_cls(x,y,pre)















# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:11:29 2022

@author: Shibdas Bhattacharya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:50:20 2022

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:33:09 2022

@author: hp
"""
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras import layers, initializers

import numpy as np
from sklearn.model_selection import KFold
tf.__version__

keras.__version__

#(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
#C:\Users\hp\Downloads
# df1 =  pd.read_excel(r"C:\Users\hp\Downloads\testdata.xlsx", sheet_name ='Sheet1')
# df2 =  pd.read_excel(r"C:\Users\hp\Downloads\testdata.xlsx", sheet_name ='Sheet2')
df1 =  pd.read_excel(r"C:\Users\hp\Downloads\Model.marketype.phase3.xlsx", sheet_name ='Sheet1')
#df2 =  pd.read_excel(r"C:\Users\hp\Downloads\testdata2.xlsx", sheet_name ='Sheet3')
#Using Pearson Correlation
y =  pd.read_excel(r"C:\Users\hp\Downloads\y.xlsx", sheet_name ='Sheet1')
plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
dataset = df1.dropna()
#dataset1=df2.dropna()
# y11=dataset1['Close']
# X1=dataset1.drop(columns=['Close', 'Unnamed: 0','ShortInt','ShortIntRat','PERatio'])
#y = dataset['marketType']
X = dataset.drop(columns=['Reporting Date', 'marketType'])

# scalers={}

# for i in X1.columns:

#     scaler = MinMaxScaler(feature_range=(0,1))

#     s_s = scaler.fit_transform(X1[i].values.reshape(-1,1) )

#     s_s=np.reshape(s_s,len(s_s))

#     scalers['scaler_'+ i] = scaler

#     X1[i]=s_s
 
scalers={}

for i in X.columns:

    scaler = MinMaxScaler(feature_range=(0,1))

    s_s = scaler.fit_transform(X[i].values.reshape(-1,1))

    s_s=np.reshape(s_s,len(s_s))

    scalers['scaler_'+ i] = scaler

    X[i]=s_s
##shifting
X=X.to_numpy()
# r=np.size(X, 0)
# r=r-1
# X=np.delete(X, r, 0)
# y1 = [None] * (len(y)-1)
# y1=a = np.array(y1).astype('float64')
# for i in range(0,np.size(y)-1):
#     y1[i]=y[i+1]
   
y=y.to_numpy()
#k-fold validation
kf = KFold(n_splits=10)
fold_no=1
#intializing score
scores=np.arange(10)
for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # #scaling y
    # scaler = MinMaxScaler(feature_range=(-1,1))
   
    # s_s = scaler.fit_transform(y.values.reshape(-1,1))
    # y=s_s
   
    # scalers={}
   
    # #scaling y11
    # scaler = MinMaxScaler(feature_range=(-1,1))
   
    # s_s = scaler.fit_transform(y11.values.reshape(-1,1))
    # y111=s_s
   
   
    #shifting the data for final test
    # X1=X1.to_numpy()
    # r=np.size(X1, 0)
    # r=r-1
    # X1=np.delete(X1, r, 0)
    # #y11=y11.to_numpy()
    # y111=y11[1:r+1]
   
   
    #shifting the data for training
   
   
    # split into train test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=101)
    #standardization scaler - fit&transform on train, fit only on test
    # print ("helo hello")
    # from sklearn.preprocessing import StandardScaler
    # s_scaler = StandardScaler()
    # X_train = s_scaler.fit_transform(X_train)
    # X_test = s_scaler.transform(X_test)
    # print ("helo hello")
    # Creating a Neural Network Model
   
        model = keras.models.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(9,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        # keras.layers.Dense(32,activation='selu',kernel_initializer="lecun_normal"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(16,activation='selu',kernel_initializer="lecun_normal"),
        # keras.layers.BatchNormalization(),
        keras.layers.Dense(64,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(9,activation='selu',kernel_initializer="lecun_normal",bias_initializer=initializers.Constant(0.5)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(8)
         
       
        ])
    # model = keras.models.Sequential([
    #     keras.layers.Dense(9,activation='elu',kernel_initializer="he_normal"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(32,activation='elu',kernel_initializer="he_normal"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(64,activation='elu',kernel_initializer="he_normal"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(32,activation='elu',kernel_initializer="he_normal"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(9,activation='elu',kernel_initializer="he_normal"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(1)
         
       
    #     ])
   
   
    #optimizer=keras.optimizers.RMSprop(lr=0.001,rho=0.9)
    #optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2= 0.999)
        #model.compile(optimizer='adam',loss='mae')
        model.compile(loss="hinge",
                     optimizer="nadam",
                      metrics=["accuracy"])
        checkpoint_cb = keras.callbacks.ModelCheckpoint("D:\\ML_project\\project_stock_closing_preciction\\reg_nn_keras_model"+str(fold_no)+".h5", save_best_only=True)
        model.fit(x=X_train,y=y_train,
                  validation_data=(X_test,y_test),
                  batch_size=20,epochs=3000,    callbacks=[checkpoint_cb])
       
       # model = keras.models.load_model("D:\\ML_project\\project_stock_closing_preciction\\reg_nn_keras_model"+str(fold_no)+".h5") # rollback to best model
        y_pred1 = model.predict(X_test)
        # Plot the data
        #plt.plot(y_test, y_pred1, label='linear')
        x = np.arange(len(y_test))
        # # y111=y111.to_numpy()
        # # y_pred1=y_pred1.to_numpy()
        plt.figure(figsize=(6,4))
        plt.plot( x,y_test, color='r', label='original')
        plt.plot( x,y_pred1[:,0], color='g', label='prediction')
       
        # # Add a legend
        plt.legend()
        # # Show the plot
        plt.show()
        # Generate generalization metrics
        model.evaluate(X_test,y_test, verbose=0)
        # scores[fold_no-1] = model.evaluate(X_test,y_test, verbose=0)
        # print('score for fold 1 is %d',scores[fold_no-1])
        #print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        # acc_per_fold.append(scores[1] * 100)
        # loss_per_fold.append(scores[0])
       
        # Increase fold number
        fold_no = fold_no + 1
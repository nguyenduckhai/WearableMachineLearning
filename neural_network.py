from util import* 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint

# Import data from database 
url_Acc_data = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/15_7_2019_acc_faster.csv"
url_Gyro_data = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/15_7_2019_gyro_faster.csv"
url_wearable_label = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/15_7_20192_wearable_labels.csv"
url_activities = "/home/khai/SENSOR_PROJECT/WearablePython/data/1_7_2019_activities.csv"

url_Acc_shake_walk = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/3_9_2019_acc.csv"
url_Gyro_shake_walk = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/3_9_2019_gyro.csv"
url_wearable_label_shake_walk = "/home/khai/SENSOR_PROJECT/WearablePython/data/Training_data/3_9_2019_wearable_labels.csv"

url_Acc_shake_walk_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/7_8_20192_acc_shake_walk_faster.csv"
url_Gyro_shake_walk_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/7_8_20192_gyro_shake__walk_faster.csv"
url_wearable_label_shake_walk_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/7_8_20192_wearable_labels_shake_walk_faster.csv"

url_Acc_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/6_8_2019_acc_faster.csv"
url_Gyro_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/6_8_2019_gyro_faster.csv"
url_wearable_label_test = "/home/khai/SENSOR_PROJECT/WearablePython/data/Testing_data/6_8_2019_wearable_labels.csv"

# Extract data from data 
df_activities =  pd.read_csv(url_activities,header=0,sep=';', names = None)
df_acc =  pd.read_csv(url_Acc_data,header=0,sep=',', names = None)
df_gyro =  pd.read_csv(url_Gyro_data,header=0,sep=',', names = None)
df_label = pd.read_csv(url_wearable_label,header=0,sep=',', names = None)

df_Acc_shake_walk_test = pd.read_csv(url_Acc_shake_walk_test,header=0,sep=',', names = None)
df_Gyro_shake_walk_test = pd.read_csv(url_Gyro_shake_walk_test,header=0,sep=',', names = None)
df_wearable_label_shake_walk_test = pd.read_csv(url_wearable_label_shake_walk_test,header=0,sep=',', names = None)

df_Acc_shake_walk = pd.read_csv(url_Acc_shake_walk,header=0,sep=';', names = None)
df_Gyro_shake_walk = pd.read_csv(url_Gyro_shake_walk,header=0,sep=';', names = None)
df_wearable_label_shake_walk = pd.read_csv(url_wearable_label_shake_walk,header=0,sep=',', names = None)

df_acc_test = pd.read_csv(url_Acc_test,header=0,sep=',', names = None)
df_gyro_test = pd.read_csv(url_Gyro_test,header=0,sep=',', names = None)
df_wearabale_test = pd.read_csv(url_wearable_label_test,header=0,sep=',', names = None)

# Extract to matrix 
X,y = load_data_faster(df_label,df_acc,df_gyro,20)
X_more, y_more = load_data_faster(df_wearable_label_shake_walk,df_Acc_shake_walk,df_Gyro_shake_walk,20) 
X_total = np.append(X,X_more,axis=0)

X_test,y_test = load_data_faster(df_wearabale_test,df_acc_test,df_gyro_test,20)
X_test_more,y_test_more = load_data_faster(df_wearable_label_shake_walk_test,df_Acc_shake_walk_test,df_Gyro_shake_walk_test,20)
X_test_total = np.append(X_test,X_test_more,axis=0)

y_total = np.append(y,y_more,axis=0)
y_test_total = np.append(y_test,y_test_more,axis=0)

# minus 1 because Neural network only except class from [0:4] not [1:5]
y_total -= 1
y_test_total -=1

X_total = np.delete(X_total,20,1)
X_test_total = np.delete(X_test_total,20,1)

# Turn into 1 hot coding
y_train = to_categorical(y_total)
y_test = to_categorical(y_test_total)

# Scale data
scaler = MinMaxScaler()
scaler.fit(X_total)

X_train = scaler.transform(X_total)
X_test = scaler.transform(X_test_total)

# Feedfroward Neural Network
lr = 1e-4
batch_size = 64
l2_weight = 1e-4

model = models.Sequential([
    layers.Dense(64, input_shape=(35,), activation='relu', kernel_regularizer=regularizers.l2(l2_weight)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu' , kernel_regularizer=regularizers.l2(l2_weight)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(l2_weight))
])

model_checkpoint = ModelCheckpoint("/home/khai/SENSOR_PROJECT/Jupyter_test/best_model_NN.h5", monitor='val_acc',save_best_only=True,
                                  mode="max")

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=200,
          validation_data=(X_test, y_test),
         callbacks=[model_checkpoint])
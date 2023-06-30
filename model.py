import cv2
import os
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split


dir_data="data_jmg"
X=[]



dict={"domixi":[1,0,0,0,0],"nhism":[0,1,0,0,0],"Phuong":[0,0,1,0,0],"ram_Bo":[0,0,0,1,0],"Thay_giao_ba":[0,0,0,0,1]}
for what in os.listdir(dir_data):
    what_phat=os.path.join(dir_data,what)
    list_file_name_path=[]
    for file_name in os.listdir(what_phat):
        file_name_path=os.path.join(what_phat,file_name)
        img=cv2.imread(file_name_path)
        label=file_name_path.split("\\")[1]
        list_file_name_path.append((img,dict[label]))
    X.extend(list_file_name_path)

np.random.shuffle(X)
np.random.shuffle(X)
np.random.shuffle(X)

X_data = np.array([x[0] for i, x in enumerate(X)])  # Chuyển dữ liệu hình ảnh thành mảng NumPy
Y_data = np.array([x[1] for i, x in enumerate(X)])  # Chuyển đổi nhãn thành mảng NumPy

# X_train_val,X_test,Y_train_val,Y_test=train_test_split(X_data,Y_data,test_size=0.2)
# X_train,X_val,Y_train,Y_val=train_test_split(X_train_val,Y_train_val,test_size=0.2) 


model_train=Sequential([
    layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    Dense(1000,activation='relu'),
    Dense(256,activation='relu'),
    Dense(5,activation='softmax')
])
model_train.summary()
model_train.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
model_train.fit(X_data,Y_data,epochs=10)
model_train.save("nhan_dien.h5")

# model=load_model("nhan_dien.h5")
# loss,acc=model.evaluate(X_test,Y_test)
# print(loss,acc)
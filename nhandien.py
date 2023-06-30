import cv2
from keras.models import load_model
import numpy as np
face_cascade = cv2.CascadeClassifier("C:/Users/Dell/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
result=["DoMixi","Nhism","Phuong","Ram-bo","Thay ba"]
model=load_model("nhan_dien.h5")
def anh(dirr):
    img=cv2.imread(dirr)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces:
        roi=cv2.resize(img[y:y+h,x:x+w],(64,64))
        kq=np.argmax(model.predict(roi.reshape(-1,64,64,3)))
        cv2.putText(img,result[kq],(x+10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
    while True:
        cv2.imshow('Face Detection',img)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def vd(dirr=0):
    cap=cv2.VideoCapture(dirr)
    while True:
        ret,fram=cap.read()
        if ret==False:
            break
        gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        for (x,y,w,h) in faces:
            roi=cv2.resize(fram[y:y+h,x:x+w],(64,64))
            kq=np.argmax(model.predict(roi.reshape(-1,64,64,3)))
            cv2.putText(fram,result[kq],(x+10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        cv2.imshow('Face Detection',fram)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break




if __name__=="__main__":
    vd(0)
    # anh("C:\\Users\\Dell\\Pictures\\Screenshots\\Screenshot 2023-06-30 213401.png")
    
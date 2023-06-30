import cv2
import os
from time import sleep
dir_path="data"
face_cascade = cv2.CascadeClassifier("C:/Users/Dell/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

class xuly:
    def __init__(self,path=0) -> None:
        self.path=path
    
    def lay_qua_anh(self,lenn):
        cap=cv2.imread(self.path)

        print(self.path)
        foder_name=self.path.split("\\")[1]
       
    
        if not os.path.isdir("data_jmg/"+foder_name):
            os.mkdir("data_jmg/"+foder_name)

        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3,5)

        for (x, y, w, h) in faces:
        
            roi=cv2.resize(cap[y:y+h,x:x+w],(64,64))

            cv2.imwrite("data_jmg\\"+foder_name+ "\\"+foder_name+str(lenn)+".jpg",roi)
        cv2.destroyAllWindows()
    def lay_quay_vd(self,lenn):
        count=0
        foder_name=self.path.split("\\")[1]
        cap=cv2.VideoCapture(self.path)
        print(self.path)
        if not os.path.isdir("data_jmg/"+foder_name):
            os.mkdir("data_jmg/"+foder_name)
           
        while True:
            ret, frame = cap.read()
            if ret==False:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
            faces = face_cascade.detectMultiScale(gray, 1.3,5)

            for (x, y, w, h) in faces:
            
                roi=cv2.resize(frame[y:y+h,x:x+w],(64,64))
                # sleep(1)
                cv2.imwrite("data_jmg\\"+foder_name+ "\\"+str(count)+"_"+str(lenn)+"_vd.jpg",roi)

                count=count+1
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.destroyAllWindows()
  


    # Hiển thị khung hình
            # cv2.imshow('Face Detection',frame)

        # Nhấn 'q' để thoát
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #      break

# def dinhdang():
    
#     for what in os.listdir(dir_path):
#         i=0
#         path=os.path.join(dir_path,what)
#         for st in os.listdir(path):
#             dirr=os.path.join(path,st)
#             dirrs=dirr.split("\\")[0]+"\\"+dirr.split("\\")[1]+"\\"+dirr.split("\\")[1]+"_"+str(i)+".jpg"
#             os.rename(dirr,dirrs)
#             i=i+1
#             print(dirr, dirrs)

# dinhdang()
def luachon(chose):
    for what in os.listdir(dir_path):
        path=os.path.join(dir_path,what)
        x=0
        for st in os.listdir(path):
            dirr=os.path.join(path,st)
            if chose==1:
                 if dirr.endswith(".jpg"):
                    xuly(dirr).lay_qua_anh(x)
            elif chose==2:
                if dirr.endswith(".mp4"):
                    xuly(dirr).lay_quay_vd(x)
                    
                  
            x=x+1


if __name__=="__main__":
    # chose=int(input("Moi ban nhap lua chon "))
    luachon(1)
    luachon(2)
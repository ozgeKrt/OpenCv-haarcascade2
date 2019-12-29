import cv2
import numpy as np

cap = cv2.VideoCapture("People - 6387.mp4")
insan_bulucu = cv2.CascadeClassifier("haarcascade_fullbody.xml")   #kullanacağımız filterı yazdık

while True:
    ret,kare = cap.read()       
    #videonun çalıştığına dair bool classından bir obje (ret) ve 
    #numpy sınıfından bir obje gönderiyor(kare)
    #numpy sınıfından bir obje demek yazacağımız saniyeler arasındaki kare sayısı olacak
    griton=cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    bedenler = insan_bulucu.detectMultiScale(griton,1.1,3)
    #insan bedenlerine ait dikdörtgenler döndürüyor.dikdirtgenin sol üst ve sağ üst köşesi

    for (x,y,w,h) in bedenler:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),3)        

    cv2.imshow("video",kare)

    if cv2.waitKey(5) & 0xff == ord('q'):
           break

cap.release()
cv2.destroyAllWindows()

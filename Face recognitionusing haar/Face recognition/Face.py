import numpy as np
import cv2
import pickle



face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture('http://192.168.3.200:4747/mjpegfeed?640x480')
#
while(True):
    #capture frame by frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #converting the images to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detecting the grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    #coordinates of faces
    for (x,y,w,h) in faces:
        #print(x,y,w,h)

        #reigon of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        #recognize? We can also use deep learning models keras, tensorflow, pytorch and scikitlearn
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


        img_item = "myimage.png"
        cv2.imwrite(img_item, roi_gray)

        #drwaing a frame on the face
        color = (255,0,0) #BGR (0,255)
        stroke = 2  #how fit do we want the line to be

        cv2.rectangle(frame, (x,y),(x+w,y+h), color, stroke)



    """Note: we are displaying the things in color
     frame even if we detecting them in gray """
    #display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done, release the captures
cap.release()
cv2.destroyAllWindows()


#1. Generate dataset
#2. Train the classifier and save it
#3. Detect the face and named it if it is already stored in our dataset

import cv2

# Generate dataset
def generate_dataset():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        #Scaling factor = 1.3
        #Minimum Neighbour = 5

        if faces is (): #if no face is detected
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0

    while True:
        ret,frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame),(400,400))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # 50,50 is the position of the text
            # 1 is the font size
            # (0,255,0) is the color of the text
            # 2 is the thickness of the text

            cv2.imshow("Cropped Face",face)
            if cv2.waitKey(1) == 13 or int(img_id) == 100:
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete! Thanks!")
generate_dataset()

# Train the classifier and save it
import numpy as np
from PIL import Image
import os

def train_classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        id = int(os.path.split(image)[1].split('.')[1])

        faces.append(imageNp)
        ids.append(id)
    
    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    print("Training Complete!")
train_classifier("data")

# Detect the face and named it if it is already stored in our dataset
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)

    coords = []

    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        id,pred = clf.predict(gray_image[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
        # 300 is the maximum value of the prediction
        # 0 is the minimum value of the prediction
        # 100 is the maximum value of the confidence
        # 0 is the minimum value of the confidence


        if confidence > 77:
            if id == 1:
                cv2.putText(img,"English",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
            if id == 2:
                cv2.putText(img,"NotEnglish",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        else:
            cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)

        coords = [x,y,w,h]
    return coords

def recognize(img,clf,faceCascade):
    coords = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)

while True:
    ret,img = video_capture.read()
    img = recognize(img,clf,faceCascade)
    cv2.imshow("Face Detection",img)
    if cv2.waitKey(1) == 13:
        break
print("Thanks for using this program!")
video_capture.release()
cv2.destroyAllWindows()

     
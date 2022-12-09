# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import csv
import string
import unicodedata
import regex as re



# Create window
window = tk.Tk()
window.title("Face Recognition System")

l1 = tk.Label(window, text="Name:", font=("Algerian", 20))
l1.grid(row=0, column=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(row=0, column=1)

l2 = tk.Label(window, text="Age:", font=("Algerian", 20))
l2.grid(row=1, column=0)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(row=1, column=1)

l3 = tk.Label(window, text="Your ID:", font=("Algerian", 20))
l3.grid(row=2, column=0)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(row=2, column=1)


# Any function that you want to call when the button is clicked
def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo("Error", "All fields are required")
    else:
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

        user = t1.get()
        age = t2.get()
        id = t3.get()

        # Check id in person.csv file if it is already exist or not if it is exist then show error message and return to the main window
        with open("PersonDetails/person.csv","r") as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if str(id) in row:
                    messagebox.showinfo("Error", "ID already exist")
                    return
        csvFile.close()

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


        # put user info in person.csv file in next row
        user = unicodedata.normalize('NFKD', user).encode('ascii','ignore').decode('utf-8')

        row = [user,str(age),str(id)]
        with open("PersonDetails/person.csv","a+") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

      
        messagebox.showinfo("Result", "Generating dataset completed")
    
def train_classifier():
    data_dir = ("data")
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
    messagebox.showinfo("Result", "Training dataset completed")

def detect_face():
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
                # Check id in person.csv file and print info of the user in this row on the screen
                with open("PersonDetails/person.csv","r") as csvFile:
                    reader = csv.reader(csvFile)
                    for row in reader:
                        if str(id) in row:
                            cv2.putText(img,f"Name: {row[0]}",(x,y-75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
                            cv2.putText(img,f"Age: {row[1]}",(x,y-55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
                            cv2.putText(img,f"ID: {row[2]}",(x,y-35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)

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
    video_capture.release()
    cv2.destroyAllWindows()

b1 = tk.Button(window, text="Generate Dataset", font=("Algerian", 20), bg="red", fg="white", command = generate_dataset)
b1.grid(row=3, column=0)

b2 = tk.Button(window, text="Training Dataset", font=("Algerian", 20), bg="green", fg="white", command = train_classifier)
b2.grid(row=3, column=1)

b3 = tk.Button(window, text="Detect", font=("Algerian", 20), bg="blue", fg="white", command = detect_face)
b3.grid(row=3, column=2)


window.geometry("900x500")
window.mainloop()

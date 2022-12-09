import tkinter as tk
from tkinter import messagebox
from tkinter import *
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import csv
import string
import unicodedata


# Create window
window = tk.Tk()
window.title("Face Recognition System")
window.resizable(0,0)

# Make header background image
header = Image.open("background.png")
header = header.resize((800, 150), Image.ANTIALIAS)
header = ImageTk.PhotoImage(header)
header_label = Label(window, image=header)
header_label.place(x=0, y=0)

# All functions
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

            if faces == (): #if no face is detected
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

def capture_image():
    def make_cap(img,classifier,scaleFactor,minNeighbors,color,text,clf):
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
                            a1 = tk.Label(canvas3, text=f"Name: {row[0]}", font=("Algerian", 20))
                            a1.place(x=5, y=230)
                            b2 = tk.Label(canvas3, text=f"Age: {row[1]}", font=("Algerian", 20))
                            b2.place(x=5, y=270)
                            c3 = tk.Label(canvas3, text=f"ID: {row[2]}", font=("Algerian", 20))
                            c3.place(x=5, y=310)

            else:
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)

            coords = [x,y,w,h]

            # Get x,y,w,h from the coords list
            x,y,w,h = coords[0],coords[1],coords[2],coords[3]
            cropped_face = img[y:y+h,x:x+w]
            cv2.imwrite("capture.jpg",cropped_face)
            load = Image.open("capture.jpg")
            photo = ImageTk.PhotoImage(load)
            imgcv3 = tk.Label(canvas3, image=photo, width=200, height=200)
            imgcv3.image = photo
            imgcv3.place(x=0, y=5)

        # Labels can be text or images
        
        return coords


    def recognize(img,clf,faceCascade):
        coords = make_cap(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
        return img
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)
    

    while True:
        ret,img = video_capture.read()
        img = recognize(img,clf,faceCascade)
        cv2.imshow("Face Capture",img)
        if cv2.waitKey(1) == 13:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    # cap = cv2.VideoCapture(0)
    # if cap.isOpened():
    #     ret, frame = cap.read()
    # else:
    #     ret = False
    
    # face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = face_classifier.detectMultiScale(gray,1.3,5)
    # if faces == (): #if no face is detected
    #     print("No face found")
    #     return None
    # for (x,y,w,h) in faces:
    #     cropped_face = img[y:y+h,x:x+w]
    # cv2.imwrite("capture.jpg",cropped_face)

   

    # cap.release()

    # clf = cv2.face.LBPHFaceRecognizer_create()
    # clf.read("classifier.xml")

    # for (x,y,w,h) in faces:
    #     id,pred = clf.predict(gray[y:y+h,x:x+w])
    #     confidence = int(100*(1-pred/300))

    #     # Check id in person.csv file and print info of the user in canvas3
    #     with open("PersonDetails/person.csv","r") as csvFile:
    #         reader = csv.reader(csvFile)
    #         for row in reader:
    #             if str(id) in row:
    #                 a1 = tk.Label(canvas3, text=f"Name: {row[0]}", font=("Algerian", 20))
    #                 a1.place(x=5, y=210)
    #                 b2 = tk.Label(canvas3, text=f"Age: {row[1]}", font=("Algerian", 20))
    #                 b2.place(x=5, y=250)
    #                 c3 = tk.Label(canvas3, text=f"ID: {row[2]}", font=("Algerian", 20))
    #                 c3.place(x=5, y=290)
        

# Make any canvas
canvas1 = Canvas(window, width=500, height=300, bg="ivory")
canvas1.place(x=5, y=160)
load1 = Image.open("canvas1.png")
img1 = ImageTk.PhotoImage(load1)
canvas1.create_image(250, 125, image=img1)

# Labels and entries of canvas1
l1 = tk.Label(canvas1, text="Name:", font=("Algerian", 20))
l1.place(x=5, y=5)
t1 = tk.Entry(canvas1, width=50, bd=5)
t1.place(x=150, y=10)

l2 = tk.Label(canvas1, text="Age:", font=("Algerian", 20))
l2.place(x=5, y=50)
t2 = tk.Entry(canvas1, width=50, bd=5)
t2.place(x=150, y=55)

l3 = tk.Label(canvas1, text="Your ID:", font=("Algerian", 20))
l3.place(x=5, y=95)
t3 = tk.Entry(canvas1, width=50, bd=5)
t3.place(x=150, y=100)

b1 = tk.Button(canvas1, text="Generate Dataset", font=("Algerian", 20), bg="pink", fg="black", command=generate_dataset)
b1.place(x=5, y=150)

b2 = tk.Button(canvas1, text="Training", font=("Algerian", 20), bg="pink", fg="black", command=train_classifier)
b2.place(x=320, y=150)

load2 = Image.open("canvas2.png")
img2 = ImageTk.PhotoImage(load2)


canvas2 = Canvas(window, width=500, height=300, bg="blue")
canvas2.place(x=5, y=470)
canvas2.create_image(250, 125, image=img2)

b3 = tk.Button(canvas2, text="Capture the image", font=("Algerian", 20), bg="pink", fg="black", command=capture_image)
b3.place(x=5, y=50)

b4 = tk.Button(canvas2, text="Predict face from live video", font=("Algerian", 20), bg="pink", fg="black", command=detect_face)
b4.place(x=5, y=150)

canvas3 = Canvas(window, width=280, height=590, bg="green")
canvas3.place(x=510, y=160)
load3 = Image.open("canvas3.png")
img3 = ImageTk.PhotoImage(load3)
canvas3.create_image(140, 295, image=img3)


window.geometry("800x800")
window.mainloop()
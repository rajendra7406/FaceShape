#importing the libraries
import numpy as np
import cv2
import dlib

imagepath = "D:\workspace\FaceShape\i8.jpg"
# link = https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade_path = "D:\workspace\FaceShape\haarcascade_frontalface_default.xml"
# download file path = http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_path = "D:\workspace\FaceShape\shape_predictor_68_face_landmarks.dat"

#create the haar cascade for detecting face
faceCascade = cv2.CascadeClassifier(cascade_path)

#create the landmark predictor
predictor = dlib.shape_predictor(predictor_path)

#read the image
image = cv2.imread(imagepath)
#convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in the image
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.05,
	minNeighbors=5,
	minSize=(100,100),
	flags=cv2.CASCADE_SCALE_IMAGE
	)

print("found {0} faces!".format(len(faces)) )

#looping through no of faces detected
for (x,y,w,h) in faces:
	#draw a rectangle around the faces
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

	#converting the opencv rectangle coordinates to Dlib rectangle
	dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
	print(dlib_rect)

	#detecting landmarks
	detected_landmarks = predictor(image, dlib_rect).parts()

	#converting to np matrix
	landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])

	#copying the image so we can we side by side
	original = image.copy()

	for idx, point in enumerate(landmarks):
		pos = (point[0,0], point[0,1] )

		#annotate the positions
		cv2.putText(original, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
           fontScale=0.4, color = (0,0,255) )

		#draw points on the landmark positions 
		cv2.circle(original, pos, 3, color=(0,255,255))

cv2.imshow("Faces found", image)
cv2.imshow("Landmarks found", original)





import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import math
from PIL import Image
from subprocess import call
import os
import array

#Basically finds distance between 2 points
def getDistance(tempshape, point1, point2):
	point1x = tempshape.part(point1).x
	point1y = tempshape.part(point1).y
	point2x = tempshape.part(point2).x
	point2y = tempshape.part(point2).y
				
	dx = (point2x - point1x) ** 2
	dy = (point2y - point1y) ** 2
	distance = math.sqrt(dx + dy)
	return distance

#Importing Haar cascade and dlib's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Start video
video = cv2.VideoCapture(0)

#Set name of new user
currentUser = raw_input("Enter name of current user : ")

#TOTAL LANDMARKS DETECTED ARE FROM 0-67
totals = []
counts = []
for i in range(0, 66):
	totals.append(0.0)
	counts.append(0)

while(True) :
	ret, frame = video.read()
	cv2.imshow('Original video feed', frame)

	#Convert the frame to grayscale
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Activating Haar cascade classifier to detect faces
	faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

	for(x, y, w, h) in faces :
		#Cropping and resizing face area
		pillowImage = Image.fromarray(frame[y:y+h, x:x+w])

		#Resizing dimensions
		resizedHeight = 300
		resizedWidth = 300
		######

		faceCropped = np.array(pillowImage.resize((resizedHeight, resizedWidth), Image.ANTIALIAS))

		#Initialize dlib's rectangle to start plotting points over shape of the face
		dlibRect = dlib.rectangle(0, 0, resizedHeight, resizedWidth)
		shape = predictor(cv2.cvtColor(faceCropped, cv2.COLOR_BGR2GRAY), dlibRect)
		shapeCopy = shape
		shape = face_utils.shape_to_np(shape)

	
		for z in range(0, 66):
			temp = getDistance(shapeCopy, 67, z)
			prevTotal = totals[z]
			prevCount = counts[z]
			totals.remove(totals[z])
			counts.remove(counts[z])
			totals.insert(z, prevTotal+temp)
			counts.insert(z, prevCount+1)

		cv2.imshow('Detected face', faceCropped)


	if(counts[0] > 100):
		break

	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

if(counts[0] > 100):
	saveFile = open('dataStore.txt', 'a')
	saveFile.write("\n%s " % currentUser)
	for x in range(0, 66):
		averageValue = totals[x]/counts[x]
		if(x % 3 == 0):
			print("\nP {}: {}\t".format(x+1, averageValue)),
		else:
			print("P {}: {}\t".format(x+1, averageValue)),
		saveFile.write("%s " % averageValue)	
	saveFile.close()


video.release()
cv2.destroyAllWindows()
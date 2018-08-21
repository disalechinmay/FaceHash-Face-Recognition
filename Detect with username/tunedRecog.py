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
import threading

# Very simeple hash generator according to the face
def findHash(ratios):
	customHash = ""
	hashChars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	for x in ratios:
		baseNumber = str(x).split('.')[0]
		customHash += hashChars[int(baseNumber)]

		decimalPortion = x % 1
		if decimalPortion < 0.5:
			customHash += '0'
		else:
			customHash += '1'

	print(customHash)

# Basically finds distance between 2 points
# Arguments:
# 	-> tempshape: DLIB's predictor which plots facial landmark points
# 	-> point1 & point2: Points between which distance is to be found out
def getDistance(tempshape, point1, point2):
	point1x = tempshape.part(point1).x
	point1y = tempshape.part(point1).y
	point2x = tempshape.part(point2).x
	point2y = tempshape.part(point2).y
				
	dx = (point2x - point1x) ** 2
	dy = (point2y - point1y) ** 2
	distance = math.sqrt(dx + dy)
	return distance


#Importing Haar cascade and DLIB's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Precision in % -> Tunes the recognizer according to our need
precision = 0.95

targetPoints = []

currentUser = raw_input("Enter name of current user : ")
#currentUser = "Chinmay"
userRatios = []
userRatiosLow = []
userRatiosHigh = []

#Read dataStore.txt
lines = [line.rstrip('\n') for line in open('dataStore.txt')]
for line in lines:
	tempList = line.split()
	if (tempList[0] == currentUser):
		for i in range(1, len(tempList)):
			userRatios.append(float(tempList[i]))
			tempCalc = float(tempList[i]) * (1-(1-precision))
			userRatiosLow.append(float(tempCalc))
			tempCalc = float(tempList[i]) * (1+(1-precision))
			userRatiosHigh.append(float(tempCalc))

if (len(userRatios) == 0):
	print("User not found!")
	exit(0)
else:
	print(userRatios)

# Read map.txt -> Can be tuned!
# map.txt holds a collection of points which will be used to recognize a face
# map.txt holds a list of pairs between which will define a set of lines to be considered by recognizer
lines = [line.rstrip('\n') for line in open('map.txt')]
for line in lines:
	tempList = line.split()
	targetPoints.append(int(tempList[0]))

# Holds number of ratios as defined by the map
totalTargets = int(len(targetPoints))

# Start video capture (webcam)
video = cv2.VideoCapture(0)

while(True):
	ret, frame = video.read()
	cv2.imshow('Original video feed', frame)

	#Convert the frame to grayscale
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Activating Haar cascade classifier to detect faces
	faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

	for(x, y, w, h) in faces :
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

		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			cv2.imshow('Detected face', faceCropped)
			os.system('clear')

			baseLine = getDistance(shapeCopy, 28, 27)
			ratios = []

			for x in targetPoints:
				currentLine = getDistance(shapeCopy, x, 27)
				currentRatio = float(currentLine)/float(baseLine)
				ratios.append(currentRatio)

			foundFlag = 0

			for x in range(0, totalTargets):
				#os.system('clear')
					
				if(ratios[x] > userRatiosLow[x]) and (ratios[x] < userRatiosHigh[x]):
					foundFlag = foundFlag + 1
					if(x % 3 == 0):
						print("\nP {}: {}\t[TRUE]\t".format(x+1, ratios[x])),
					else:
						print("P {}: {}\t[TRUE]\t".format(x+1, ratios[x])),					
				else:
					if(x % 3 == 0):
						print("\nP {}: {}\t\t".format(x+1, ratios[x])),
					else:
						print("P {}: {}\t\t".format(x+1, ratios[x])),

					break

			if (foundFlag == totalTargets):
				os.system('clear')
				print("MATCH FOUND")
				findHash(ratios)
				exit(0)

	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

video.release()
cv2.destroyAllWindows()